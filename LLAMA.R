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
# LLAMA VALIDATION
# Ground truth: Result
# Prediction: LLAMA Result
# ============================================================

library(dplyr)
library(stringr)
library(ggplot2)

# ------------------------------------------------------------
# Define final labels
# ------------------------------------------------------------

final_labels <- c("book", "no book", "wrong", "misc")

# ------------------------------------------------------------
# Prepare LLAMA validation dataset
# ------------------------------------------------------------

llama_eval <- results %>%
  mutate(
    Truth = str_squish(str_to_lower(Result)),
    LLAMA_raw = str_squish(str_to_lower(`LLAMA Result`))
  ) %>%
  filter(
    !is.na(Truth),
    Truth %in% final_labels
  ) %>%
  mutate(
    LLAMA_valid = if_else(
      LLAMA_raw %in% final_labels,
      LLAMA_raw,
      "invalid"
    )
  )

# ------------------------------------------------------------
# Raw output distribution
# ------------------------------------------------------------

llama_raw_output_distribution <- llama_eval %>%
  count(LLAMA_raw, sort = TRUE) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 2)
  )

llama_raw_output_distribution

# ------------------------------------------------------------
# Inflation ratio
# ------------------------------------------------------------

llama_inflation_ratio <- llama_eval %>%
  summarise(
    total_predictions = n(),
    invalid_predictions = sum(LLAMA_valid == "invalid"),
    inflation_ratio = invalid_predictions / total_predictions,
    inflation_ratio_percentage = round(inflation_ratio * 100, 2)
  )

llama_inflation_ratio

# ------------------------------------------------------------
# Keep only valid predictions for standard metrics
# ------------------------------------------------------------

llama_eval_valid <- llama_eval %>%
  filter(LLAMA_valid %in% final_labels) %>%
  mutate(
    Truth = factor(Truth, levels = final_labels),
    LLAMA_valid = factor(LLAMA_valid, levels = final_labels)
  )

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

llama_confusion_matrix <- table(
  Truth = llama_eval_valid$Truth,
  Prediction = llama_eval_valid$LLAMA_valid
)

llama_confusion_matrix

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

llama_accuracy <- sum(diag(llama_confusion_matrix)) / sum(llama_confusion_matrix)

llama_accuracy

# ------------------------------------------------------------
# Precision, recall and F1-score by class
# ------------------------------------------------------------

llama_class_metrics <- lapply(final_labels, function(label) {
  
  TP <- llama_confusion_matrix[label, label]
  FP <- sum(llama_confusion_matrix[, label]) - TP
  FN <- sum(llama_confusion_matrix[label, ]) - TP
  
  precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
  recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
  f1_score <- ifelse(
    is.na(precision) | is.na(recall) | precision + recall == 0,
    NA,
    2 * precision * recall / (precision + recall)
  )
  
  data.frame(
    Class = label,
    Support = sum(llama_confusion_matrix[label, ]),
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

llama_class_metrics

# ------------------------------------------------------------
# Global validation summary
# ------------------------------------------------------------

llama_validation_summary <- data.frame(
  Model = "LLAMA",
  Total_cases = nrow(llama_eval),
  Valid_predictions = nrow(llama_eval_valid),
  Invalid_predictions = llama_inflation_ratio$invalid_predictions,
  Inflation_ratio = llama_inflation_ratio$inflation_ratio_percentage,
  Accuracy = round(llama_accuracy * 100, 2),
  Macro_precision = round(mean(llama_class_metrics$Precision, na.rm = TRUE), 2),
  Macro_recall = round(mean(llama_class_metrics$Recall, na.rm = TRUE), 2),
  Macro_F1 = round(mean(llama_class_metrics$F1_score, na.rm = TRUE), 2)
)

llama_validation_summary

# ------------------------------------------------------------
# Confusion matrix plot
# ------------------------------------------------------------

llama_confusion_df <- as.data.frame(llama_confusion_matrix)

ggplot(llama_confusion_df, aes(x = Prediction, y = Truth, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 5) +
  labs(
    title = "LLAMA Confusion Matrix",
    x = "LLAMA prediction",
    y = "Ground truth"
  ) +
  theme_minimal()

# ------------------------------------------------------------
# Per-class F1-score plot
# ------------------------------------------------------------

ggplot(llama_class_metrics, aes(x = Class, y = F1_score)) +
  geom_col() +
  labs(
    title = "LLAMA F1-score by class",
    x = "Class",
    y = "F1-score (%)"
  ) +
  theme_minimal()

