library(readxl)
library(dplyr)
library(ggplot2)
library(lubridate)
library(scales)
library(slider)
library(forecast)

setwd("~/Desktop/TFG/CODE")

results<- read_csv("call_data_rw.csv")

# ============================================================
# GEMINI VALIDATION
# Ground truth: Result
# Prediction: Gemini Result
# ============================================================

library(dplyr)
library(stringr)
library(ggplot2)

# ------------------------------------------------------------
# Define final labels
# ------------------------------------------------------------

final_labels <- c("book", "no book", "wrong", "misc")

# ------------------------------------------------------------
# Prepare Gemini validation dataset
# ------------------------------------------------------------

gemini_eval <- results %>%
  mutate(
    Truth = str_squish(str_to_lower(Result)),
    Gemini_raw = str_squish(str_to_lower(`Gemini Result`))
  ) %>%
  filter(
    !is.na(Truth),
    Truth %in% final_labels
  ) %>%
  mutate(
    Gemini_valid = if_else(
      Gemini_raw %in% final_labels,
      Gemini_raw,
      "invalid"
    )
  )

# ------------------------------------------------------------
# Raw output distribution
# ------------------------------------------------------------

gemini_raw_output_distribution <- gemini_eval %>%
  count(Gemini_raw, sort = TRUE) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 2)
  )

gemini_raw_output_distribution

# ------------------------------------------------------------
# Inflation ratio
# ------------------------------------------------------------

gemini_inflation_ratio <- gemini_eval %>%
  summarise(
    total_predictions = n(),
    invalid_predictions = sum(Gemini_valid == "invalid"),
    inflation_ratio = invalid_predictions / total_predictions,
    inflation_ratio_percentage = round(inflation_ratio * 100, 2)
  )

gemini_inflation_ratio

# ------------------------------------------------------------
# Keep only valid predictions for standard metrics
# ------------------------------------------------------------

gemini_eval_valid <- gemini_eval %>%
  filter(Gemini_valid %in% final_labels) %>%
  mutate(
    Truth = factor(Truth, levels = final_labels),
    Gemini_valid = factor(Gemini_valid, levels = final_labels)
  )

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

gemini_confusion_matrix <- table(
  Truth = gemini_eval_valid$Truth,
  Prediction = gemini_eval_valid$Gemini_valid
)

gemini_confusion_matrix

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

gemini_accuracy <- sum(diag(gemini_confusion_matrix)) / sum(gemini_confusion_matrix)

gemini_accuracy

# ------------------------------------------------------------
# Precision, recall and F1-score by class
# ------------------------------------------------------------

gemini_class_metrics <- lapply(final_labels, function(label) {
  
  TP <- gemini_confusion_matrix[label, label]
  FP <- sum(gemini_confusion_matrix[, label]) - TP
  FN <- sum(gemini_confusion_matrix[label, ]) - TP
  
  precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
  recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
  f1_score <- ifelse(
    is.na(precision) | is.na(recall) | precision + recall == 0,
    NA,
    2 * precision * recall / (precision + recall)
  )
  
  data.frame(
    Class = label,
    Support = sum(gemini_confusion_matrix[label, ]),
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

gemini_class_metrics

# ------------------------------------------------------------
# Global validation summary
# ------------------------------------------------------------

gemini_validation_summary <- data.frame(
  Model = "Gemini",
  Total_cases = nrow(gemini_eval),
  Valid_predictions = nrow(gemini_eval_valid),
  Invalid_predictions = gemini_inflation_ratio$invalid_predictions,
  Inflation_ratio = gemini_inflation_ratio$inflation_ratio_percentage,
  Accuracy = round(gemini_accuracy * 100, 2),
  Macro_precision = round(mean(gemini_class_metrics$Precision, na.rm = TRUE), 2),
  Macro_recall = round(mean(gemini_class_metrics$Recall, na.rm = TRUE), 2),
  Macro_F1 = round(mean(gemini_class_metrics$F1_score, na.rm = TRUE), 2)
)

gemini_validation_summary

# ------------------------------------------------------------
# Confusion matrix plot
# ------------------------------------------------------------

gemini_confusion_df <- as.data.frame(gemini_confusion_matrix)

ggplot(gemini_confusion_df, aes(x = Prediction, y = Truth, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 5) +
  labs(
    title = "Gemini Confusion Matrix",
    x = "Gemini prediction",
    y = "Ground truth"
  ) +
  theme_minimal()

