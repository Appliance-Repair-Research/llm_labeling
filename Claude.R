library(dplyr)
library(stringr)
library(ggplot2)


setwd("~/Desktop/TFG/CODE")

results<- read_csv("call_data_rw.csv")

final_labels <- c("book", "no book", "wrong", "misc")

claude_eval <- results %>%
  mutate(
    Truth = str_squish(str_to_lower(Result)),
    Claude_raw = str_squish(str_to_lower(`Claude Result`))
  ) %>%
  filter(
    !is.na(Truth),
    Truth %in% final_labels
  ) %>%
  mutate(
    Claude_valid = if_else(
      Claude_raw %in% final_labels,
      Claude_raw,
      "invalid"
    )
  )

claude_raw_output_distribution <- claude_eval %>%
  count(Claude_raw, sort = TRUE) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 2)
  )

claude_raw_output_distribution

claude_inflation_ratio <- claude_eval %>%
  summarise(
    total_predictions = n(),
    invalid_predictions = sum(Claude_valid == "invalid"),
    inflation_ratio = invalid_predictions / total_predictions,
    inflation_ratio_percentage = round(inflation_ratio * 100, 2)
  )

claude_inflation_ratio

claude_eval_valid <- claude_eval %>%
  mutate(
    Truth = factor(Truth, levels = final_labels),
    Claude_valid = factor(Claude_valid, levels = c(final_labels, "invalid"))
  )

claude_confusion_matrix <- table(
  Truth = claude_eval_valid$Truth,
  Prediction = claude_eval_valid$Claude_valid
)

claude_confusion_matrix

claude_accuracy <- sum(
  as.character(claude_eval_valid$Truth) == as.character(claude_eval_valid$Claude_valid),
  na.rm = TRUE
) / nrow(claude_eval_valid)


claude_accuracy

claude_class_metrics <- lapply(final_labels, function(label) {
  
  TP <- sum(claude_eval_valid$Truth == label & claude_eval_valid$Claude_valid == label, na.rm = TRUE)
  FP <- sum(claude_eval_valid$Truth != label & claude_eval_valid$Claude_valid == label, na.rm = TRUE)
  FN <- sum(claude_eval_valid$Truth == label & claude_eval_valid$Claude_valid != label, na.rm = TRUE)
  
  precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
  recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
  f1_score <- ifelse(
    is.na(precision) | is.na(recall) | precision + recall == 0,
    NA,
    2 * precision * recall / (precision + recall)
  )
  
  data.frame(
    Class = label,
    Support = sum(claude_eval_valid$Truth == label, na.rm = TRUE),
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

claude_class_metrics

claude_validation_summary <- data.frame(
  Model = "Claude",
  Data_type = "all data",
  Total_observations = nrow(claude_eval),
  Valid_predictions = sum(claude_eval$Claude_valid %in% final_labels),
  Invalid_predictions = claude_inflation_ratio$invalid_predictions,
  Inflation_rate = claude_inflation_ratio$inflation_ratio_percentage,
  Accuracy = round(claude_accuracy * 100, 2),
  Macro_precision = round(mean(claude_class_metrics$Precision, na.rm = TRUE), 2),
  Macro_recall = round(mean(claude_class_metrics$Recall, na.rm = TRUE), 2),
  Macro_F1 = round(mean(claude_class_metrics$F1_score, na.rm = TRUE), 2)
)

claude_validation_summary

claude_confusion_df <- as.data.frame(claude_confusion_matrix)

ggplot(claude_confusion_df, aes(x = Prediction, y = Truth, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 5) +
  labs(
    title = "Claude Confusion Matrix",
    x = "Claude prediction",
    y = "Ground truth"
  ) +
  theme_minimal()

