# ============================================================
# MODEL VALIDATION - ALL MODELS
# Ground truth: Result
# Predictions: Agent, Gemini, Gemini Tuned, GPT, LLAMA, LR, RoBERTa
# ============================================================

library(readr)
library(dplyr)
library(stringr)
library(ggplot2)
library(purrr)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Change this path only if needed
setwd("~/Desktop/TFG/CODE")

input_file <- "call_data_rw.csv"
final_labels <- c("book", "no book", "wrong", "misc")

model_specs <- tibble::tribble(
  ~model_name,             ~prediction_col,
  "Agent Result",          "Agent Result",
  "Gemini",                "Gemini Result",
  "Gemini Tuned",          "Gemini Tuned Result",
  "GPT",                   "GPT Result",
  "LLAMA",                 "LLAMA Result",
  "Logistic Regression",   "LR Result",
  "RoBERTa",               "Roberta Result"
)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

results <- read_csv(input_file, show_col_types = FALSE)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

clean_label <- function(x) {
  str_squish(str_to_lower(as.character(x)))
}

validate_model <- function(data, model_name, prediction_col, labels = final_labels) {
  
  if (!prediction_col %in% names(data)) {
    stop(paste("Column not found:", prediction_col))
  }
  
  eval_data <- data %>%
    mutate(
      Truth = clean_label(Result),
      Prediction_raw = clean_label(.data[[prediction_col]])
    ) %>%
    filter(
      !is.na(Truth),
      Truth %in% labels
    ) %>%
    mutate(
      Prediction_valid = if_else(
        Prediction_raw %in% labels,
        Prediction_raw,
        "invalid"
      )
    )
  
  raw_output_distribution <- eval_data %>%
    count(Prediction_raw, sort = TRUE) %>%
    mutate(
      Model = model_name,
      percentage = round(n / sum(n) * 100, 2)
    ) %>%
    select(Model, Prediction_raw, n, percentage)
  
  inflation_ratio <- eval_data %>%
    summarise(
      Model = model_name,
      total_predictions = n(),
      invalid_predictions = sum(Prediction_valid == "invalid"),
      inflation_ratio = invalid_predictions / total_predictions,
      inflation_ratio_percentage = round(inflation_ratio * 100, 2)
    )
  
  eval_data_valid <- eval_data %>%
    filter(Prediction_valid %in% labels) %>%
    mutate(
      Truth = factor(Truth, levels = labels),
      Prediction_valid = factor(Prediction_valid, levels = labels)
    )
  
  confusion_matrix <- table(
    Truth = eval_data_valid$Truth,
    Prediction = eval_data_valid$Prediction_valid
  )
  
  accuracy <- ifelse(
    sum(confusion_matrix) == 0,
    NA,
    sum(diag(confusion_matrix)) / sum(confusion_matrix)
  )
  
  class_metrics <- map_dfr(labels, function(label) {
    TP <- confusion_matrix[label, label]
    FP <- sum(confusion_matrix[, label]) - TP
    FN <- sum(confusion_matrix[label, ]) - TP
    
    precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
    recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
    f1_score <- ifelse(
      is.na(precision) | is.na(recall) | precision + recall == 0,
      NA,
      2 * precision * recall / (precision + recall)
    )
    
    data.frame(
      Model = model_name,
      Class = label,
      Support = sum(confusion_matrix[label, ]),
      True_positive = TP,
      False_positive = FP,
      False_negative = FN,
      Precision = round(precision * 100, 2),
      Recall = round(recall * 100, 2),
      F1_score = round(f1_score * 100, 2)
    )
  })
  
  validation_summary <- data.frame(
    Model = model_name,
    Total_cases = nrow(eval_data),
    Valid_predictions = nrow(eval_data_valid),
    Invalid_predictions = inflation_ratio$invalid_predictions,
    Inflation_ratio = inflation_ratio$inflation_ratio_percentage,
    Accuracy = round(accuracy * 100, 2),
    Macro_precision = round(mean(class_metrics$Precision, na.rm = TRUE), 2),
    Macro_recall = round(mean(class_metrics$Recall, na.rm = TRUE), 2),
    Macro_F1 = round(mean(class_metrics$F1_score, na.rm = TRUE), 2)
  )
  
  confusion_df <- as.data.frame(confusion_matrix) %>%
    mutate(Model = model_name) %>%
    select(Model, Truth, Prediction, Freq)
  
  list(
    model_name = model_name,
    eval_data = eval_data,
    eval_data_valid = eval_data_valid,
    raw_output_distribution = raw_output_distribution,
    inflation_ratio = inflation_ratio,
    confusion_matrix = confusion_matrix,
    confusion_df = confusion_df,
    accuracy = accuracy,
    class_metrics = class_metrics,
    validation_summary = validation_summary
  )
}

plot_confusion_matrix <- function(model_result) {
  ggplot(model_result$confusion_df, aes(x = Prediction, y = Truth, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), size = 5) +
    labs(
      title = paste(model_result$model_name, "Confusion Matrix"),
      x = paste(model_result$model_name, "prediction"),
      y = "Ground truth"
    ) +
    theme_minimal()
}

plot_f1_by_class <- function(class_metrics_all) {
  ggplot(class_metrics_all, aes(x = Class, y = F1_score, fill = Model)) +
    geom_col(position = "dodge") +
    labs(
      title = "F1-score by class and model",
      x = "Class",
      y = "F1-score (%)"
    ) +
    theme_minimal()
}

# ------------------------------------------------------------
# Run validation for all models
# ------------------------------------------------------------

model_results <- pmap(
  model_specs,
  ~ validate_model(
    data = results,
    model_name = ..1,
    prediction_col = ..2,
    labels = final_labels
  )
)

names(model_results) <- model_specs$model_name

# ------------------------------------------------------------
# Combined outputs
# ------------------------------------------------------------

validation_summary_all <- map_dfr(model_results, "validation_summary") %>%
  arrange(desc(Macro_F1))

class_metrics_all <- map_dfr(model_results, "class_metrics")

raw_output_distribution_all <- map_dfr(model_results, "raw_output_distribution")

inflation_ratio_all <- map_dfr(model_results, "inflation_ratio")

confusion_df_all <- map_dfr(model_results, "confusion_df")

# Print main tables
validation_summary_all
class_metrics_all
raw_output_distribution_all
inflation_ratio_all

# ------------------------------------------------------------
# Individual confusion matrix plots
# Examples:
# plot_confusion_matrix(model_results[["Gemini"]])
# plot_confusion_matrix(model_results[["GPT"]])
# ------------------------------------------------------------

confusion_plots <- map(model_results, plot_confusion_matrix)

# To display one plot, run for example:
confusion_plots[["Gemini"]]

# ------------------------------------------------------------
# Global F1-score comparison plot
# ------------------------------------------------------------

f1_plot <- plot_f1_by_class(class_metrics_all)
f1_plot

# ------------------------------------------------------------
# Optional: export results
# ------------------------------------------------------------

write_csv(validation_summary_all, "validation_summary_all.csv")
write_csv(class_metrics_all, "class_metrics_all.csv")
write_csv(raw_output_distribution_all, "raw_output_distribution_all.csv")
write_csv(inflation_ratio_all, "inflation_ratio_all.csv")
write_csv(confusion_df_all, "confusion_matrices_all.csv")
