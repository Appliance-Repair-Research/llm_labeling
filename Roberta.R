
library(readxl)
library(dplyr)
library(ggplot2)
library(lubridate)
library(scales)
library(slider)
library(forecast)

setwd("~/Desktop/TFG/CODE")

results<- read_csv("call_data_rw.csv")


library(dplyr)
library(stringr)
library(ggplot2)

final_labels <- c("book", "no book", "wrong", "misc")

roberta_eval <- results %>%
  mutate(
    Truth = str_squish(str_to_lower(Result)),
    Split = str_squish(str_to_lower(split)),
    Roberta_raw = str_squish(str_to_lower(`Roberta Result`))
  ) %>%
  mutate(
    Roberta_valid = if_else(
      Roberta_raw %in% final_labels,
      Roberta_raw,
      "invalid"
    )
  )

roberta_raw_output_distribution <- roberta_eval %>%
  count(Roberta_raw, sort = TRUE) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 2)
  )

roberta_raw_output_distribution

roberta_inflation_ratio <- roberta_eval %>%
  summarise(
    total_predictions = n(),
    invalid_predictions = sum(Roberta_valid == "invalid"),
    inflation_ratio = invalid_predictions / total_predictions,
    inflation_ratio_percentage = round(inflation_ratio * 100, 2)
  )

roberta_inflation_ratio

roberta_eval_valid <- roberta_eval %>%
  filter(Roberta_valid %in% final_labels) %>%
  mutate(
    Truth = factor(Truth, levels = final_labels),
    Roberta_valid = factor(Roberta_valid, levels = final_labels)
  )

roberta_confusion_matrix <- table(
  Truth = roberta_eval_valid$Truth,
  Prediction = roberta_eval_valid$Roberta_valid
)

roberta_confusion_matrix

roberta_accuracy <- sum(diag(roberta_confusion_matrix)) / sum(roberta_confusion_matrix)

roberta_accuracy

roberta_class_metrics <- lapply(final_labels, function(label) {
  
  TP <- roberta_confusion_matrix[label, label]
  FP <- sum(roberta_confusion_matrix[, label]) - TP
  FN <- sum(roberta_confusion_matrix[label, ]) - TP
  
  precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
  recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
  f1_score <- ifelse(
    is.na(precision) | is.na(recall) | precision + recall == 0,
    NA,
    2 * precision * recall / (precision + recall)
  )
  
  data.frame(
    Class = label,
    Support = sum(roberta_confusion_matrix[label, ]),
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

roberta_class_metrics

roberta_validation_summary <- data.frame(
  Model = "RoBERTa",
  Total_cases = nrow(roberta_eval),
  Valid_predictions = nrow(roberta_eval_valid),
  Invalid_predictions = roberta_inflation_ratio$invalid_predictions,
  Inflation_ratio = roberta_inflation_ratio$inflation_ratio_percentage,
  Accuracy = round(roberta_accuracy * 100, 2),
  Macro_precision = round(mean(roberta_class_metrics$Precision, na.rm = TRUE), 2),
  Macro_recall = round(mean(roberta_class_metrics$Recall, na.rm = TRUE), 2),
  Macro_F1 = round(mean(roberta_class_metrics$F1_score, na.rm = TRUE), 2)
)

roberta_validation_summary

roberta_confusion_df <- as.data.frame(roberta_confusion_matrix)

ggplot(roberta_confusion_df, aes(x = Prediction, y = Truth, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 5) +
  labs(
    title = "RoBERTa Confusion Matrix",
    x = "RoBERTa prediction",
    y = "Ground truth"
  ) +
  theme_minimal()