library(dplyr)
library(stringr)
library(ggplot2)

setwd("~/Desktop/TFG/CODE")

results<- read_csv("call_data_rw.csv")


final_labels <- c("book", "no book", "wrong", "misc")

gemini_tuned_eval <- results %>%
  mutate(
    Truth = str_squish(str_to_lower(Result)),
    Gemini_tuned_raw = str_squish(str_to_lower(`Gemini Tuned Result`))
  ) %>%
  filter(
    !is.na(Truth),
    Truth %in% final_labels
  ) %>%
  mutate(
    Gemini_tuned_valid = if_else(
      Gemini_tuned_raw %in% final_labels,
      Gemini_tuned_raw,
      "invalid"
    )
  )

gemini_tuned_raw_output_distribution <- gemini_tuned_eval %>%
  count(Gemini_tuned_raw, sort = TRUE) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 2)
  )

gemini_tuned_raw_output_distribution

gemini_tuned_inflation_ratio <- gemini_tuned_eval %>%
  summarise(
    total_predictions = n(),
    invalid_predictions = sum(Gemini_tuned_valid == "invalid"),
    inflation_ratio = invalid_predictions / total_predictions,
    inflation_ratio_percentage = round(inflation_ratio * 100, 2)
  )

gemini_tuned_inflation_ratio

gemini_tuned_eval_valid <- gemini_tuned_eval %>%
  filter(Gemini_tuned_valid %in% final_labels) %>%
  mutate(
    Truth = factor(Truth, levels = final_labels),
    Gemini_tuned_valid = factor(Gemini_tuned_valid, levels = final_labels)
  )

gemini_tuned_confusion_matrix <- table(
  Truth = gemini_tuned_eval_valid$Truth,
  Prediction = gemini_tuned_eval_valid$Gemini_tuned_valid
)

gemini_tuned_confusion_matrix

gemini_tuned_accuracy <- sum(diag(gemini_tuned_confusion_matrix)) / sum(gemini_tuned_confusion_matrix)

gemini_tuned_accuracy

gemini_tuned_class_metrics <- lapply(final_labels, function(label) {
  
  TP <- gemini_tuned_confusion_matrix[label, label]
  FP <- sum(gemini_tuned_confusion_matrix[, label]) - TP
  FN <- sum(gemini_tuned_confusion_matrix[label, ]) - TP
  
  precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
  recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
  f1_score <- ifelse(
    is.na(precision) | is.na(recall) | precision + recall == 0,
    NA,
    2 * precision * recall / (precision + recall)
  )
  
  data.frame(
    Class = label,
    Support = sum(gemini_tuned_confusion_matrix[label, ]),
    True_positive = TP,
    False_positive = FP,
    False_negative = FN,
    Precision = precision,
    Recall = recall,
    F1_score = f1_score
  )
}) %>%
  bind_rows() %>%
  mutate(
    Precision = round(Precision * 100, 2),
    Recall = round(Recall * 100, 2),
    F1_score = round(F1_score * 100, 2)
  )

gemini_tuned_class_metrics

gemini_tuned_validation_summary <- data.frame(
  Model = "Gemini Tuned",
  Total_cases = nrow(gemini_tuned_eval),
  Valid_predictions = nrow(gemini_tuned_eval_valid),
  Invalid_predictions = gemini_tuned_inflation_ratio$invalid_predictions,
  Inflation_ratio = gemini_tuned_inflation_ratio$inflation_ratio_percentage,
  Accuracy = round(gemini_tuned_accuracy * 100, 2),
  Macro_precision = round(mean(gemini_tuned_class_metrics$Precision, na.rm = TRUE), 2),
  Macro_recall = round(mean(gemini_tuned_class_metrics$Recall, na.rm = TRUE), 2),
  Macro_F1 = round(mean(gemini_tuned_class_metrics$F1_score, na.rm = TRUE), 2)
)

gemini_tuned_validation_summary

gemini_tuned_confusion_df <- as.data.frame(gemini_tuned_confusion_matrix)

ggplot(gemini_tuned_confusion_df, aes(x = Prediction, y = Truth, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 5) +
  labs(
    title = "Gemini Tuned Confusion Matrix",
    x = "Gemini tuned prediction",
    y = "Ground truth"
  ) +
  theme_minimal()