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
# GPT VALIDATION
# Ground truth: Result
# Prediction: GPT Result
# ============================================================

library(dplyr)
library(stringr)
library(ggplot2)

# ------------------------------------------------------------
# Define final labels
# ------------------------------------------------------------

final_labels <- c("book", "no book", "wrong", "misc")

# ------------------------------------------------------------
# Prepare GPT validation dataset
# ------------------------------------------------------------

gpt_eval <- results %>%
  mutate(
    Truth = str_squish(str_to_lower(Result)),
    GPT_raw = str_squish(str_to_lower(`GPT Result`))
  ) %>%
  filter(
    !is.na(Truth),
    Truth %in% final_labels
  ) %>%
  mutate(
    GPT_valid = if_else(
      GPT_raw %in% final_labels,
      GPT_raw,
      "invalid"
    )
  )

# ------------------------------------------------------------
# Raw output distribution
# ------------------------------------------------------------

gpt_raw_output_distribution <- gpt_eval %>%
  count(GPT_raw, sort = TRUE) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 2)
  )

gpt_raw_output_distribution

# ------------------------------------------------------------
# Inflation ratio
# ------------------------------------------------------------

gpt_inflation_ratio <- gpt_eval %>%
  summarise(
    total_predictions = n(),
    invalid_predictions = sum(GPT_valid == "invalid"),
    inflation_ratio = invalid_predictions / total_predictions,
    inflation_ratio_percentage = round(inflation_ratio * 100, 2)
  )

gpt_inflation_ratio

# ------------------------------------------------------------
# Keep only valid predictions for standard metrics
# ------------------------------------------------------------

gpt_eval_valid <- gpt_eval %>%
  filter(GPT_valid %in% final_labels) %>%
  mutate(
    Truth = factor(Truth, levels = final_labels),
    GPT_valid = factor(GPT_valid, levels = final_labels)
  )

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

gpt_confusion_matrix <- table(
  Truth = gpt_eval_valid$Truth,
  Prediction = gpt_eval_valid$GPT_valid
)

gpt_confusion_matrix

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

gpt_accuracy <- sum(diag(gpt_confusion_matrix)) / sum(gpt_confusion_matrix)

gpt_accuracy

# ------------------------------------------------------------
# Precision, recall and F1-score by class
# ------------------------------------------------------------

gpt_class_metrics <- lapply(final_labels, function(label) {
  
  TP <- gpt_confusion_matrix[label, label]
  FP <- sum(gpt_confusion_matrix[, label]) - TP
  FN <- sum(gpt_confusion_matrix[label, ]) - TP
  
  precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
  recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
  f1_score <- ifelse(
    is.na(precision) | is.na(recall) | precision + recall == 0,
    NA,
    2 * precision * recall / (precision + recall)
  )
  
  data.frame(
    Class = label,
    Support = sum(gpt_confusion_matrix[label, ]),
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

gpt_class_metrics

# ------------------------------------------------------------
# Global validation summary
# ------------------------------------------------------------

gpt_validation_summary <- data.frame(
  Model = "GPT",
  Total_cases = nrow(gpt_eval),
  Valid_predictions = nrow(gpt_eval_valid),
  Invalid_predictions = gpt_inflation_ratio$invalid_predictions,
  Inflation_ratio = gpt_inflation_ratio$inflation_ratio_percentage,
  Accuracy = round(gpt_accuracy * 100, 2),
  Macro_precision = round(mean(gpt_class_metrics$Precision, na.rm = TRUE), 2),
  Macro_recall = round(mean(gpt_class_metrics$Recall, na.rm = TRUE), 2),
  Macro_F1 = round(mean(gpt_class_metrics$F1_score, na.rm = TRUE), 2)
)

gpt_validation_summary

# ------------------------------------------------------------
# Confusion matrix plot
# ------------------------------------------------------------

gpt_confusion_df <- as.data.frame(gpt_confusion_matrix)

ggplot(gpt_confusion_df, aes(x = Prediction, y = Truth, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 5) +
  labs(
    title = "GPT Confusion Matrix",
    x = "GPT prediction",
    y = "Ground truth"
  ) +
  theme_minimal()

