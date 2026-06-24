# ============================================================
# MODEL COMPARISON TABLE
# Summary comparison across all LLM models
# ============================================================
library(dplyr)
library(knitr)
library(stringr)
library(purrr)
library(tidyr)

final_labels <- c("book", "no book", "wrong", "misc")

setwd("~/Desktop/TFG/CODE")

results<- read_csv("call_data_rw.csv")


# Function to evaluate one model on the test split
evaluate_model_test <- function(data, pred_col, model_name) {
  
  eval_data <- data %>%
    mutate(
      Truth = str_squish(str_to_lower(Result)),
      Prediction = str_squish(str_to_lower(.data[[pred_col]])),
      Split = str_squish(str_to_lower(split))
    ) %>%
    filter(
      Split == "test",
      !is.na(Truth),
      Truth %in% final_labels
    )
  
  total_obs <- nrow(eval_data)
  
  accuracy <- mean(eval_data$Truth == eval_data$Prediction, na.rm = TRUE)
  
  class_metrics <- map_dfr(final_labels, function(label) {
    
    TP <- sum(eval_data$Truth == label & eval_data$Prediction == label, na.rm = TRUE)
    FP <- sum(eval_data$Truth != label & eval_data$Prediction == label, na.rm = TRUE)
    FN <- sum(eval_data$Truth == label & eval_data$Prediction != label, na.rm = TRUE)
    
    precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
    recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
    f1 <- ifelse(
      is.na(precision) | is.na(recall) | precision + recall == 0,
      NA,
      2 * precision * recall / (precision + recall)
    )
    
    data.frame(
      Class = label,
      Precision = precision,
      Recall = recall,
      F1 = f1
    )
  })
  
  data.frame(
    Model = model_name,
    Data_type = "test",
    Total_observations = total_obs,
    Accuracy = round(accuracy * 100, 2),
    Precision = round(mean(class_metrics$Precision, na.rm = TRUE) * 100, 2),
    Recall = round(mean(class_metrics$Recall, na.rm = TRUE) * 100, 2),
    F1_score = round(mean(class_metrics$F1, na.rm = TRUE) * 100, 2)
  )
}

model_performance_table <- bind_rows(
  evaluate_model_test(results, "Agent Result", "Agent"),
  evaluate_model_test(results, "LR Result", "Logistic Regression"),
  evaluate_model_test(results, "Roberta Result", "RoBERTa"),
  evaluate_model_test(results, "GPT Result", "GPT"),
  evaluate_model_test(results, "Gemini Result", "Gemini"),
  evaluate_model_test(results, "LLAMA Result", "LLaMA"),
  evaluate_model_test(results, "Claude Result", "Claude"),
  evaluate_model_test(results, "Gemini Tuned Result", "Gemini Tuned")
  
)

model_performance_table
##------------------------------------##

final_labels <- c("book", "no book", "wrong", "misc")

evaluate_llm_all_data_inflation <- function(data, pred_col, model_name) {
  
  eval_data <- data %>%
    mutate(
      Truth = str_squish(str_to_lower(Result)),
      Prediction = str_squish(str_to_lower(.data[[pred_col]]))
    ) %>%
    filter(
      !is.na(Truth),
      Truth %in% final_labels
    ) %>%
    mutate(
      Prediction_valid = if_else(
        Prediction %in% final_labels,
        Prediction,
        "invalid"
      )
    )
  
  total_obs <- nrow(eval_data)
  
  invalid_predictions <- sum(eval_data$Prediction_valid == "invalid", na.rm = TRUE)
  
  inflation_rate <- invalid_predictions / total_obs
  
  accuracy <- mean(eval_data$Truth == eval_data$Prediction_valid, na.rm = TRUE)
  
  class_metrics <- map_dfr(final_labels, function(label) {
    
    TP <- sum(eval_data$Truth == label & eval_data$Prediction_valid == label, na.rm = TRUE)
    FP <- sum(eval_data$Truth != label & eval_data$Prediction_valid == label, na.rm = TRUE)
    FN <- sum(eval_data$Truth == label & eval_data$Prediction_valid != label, na.rm = TRUE)
    
    precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
    recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
    f1 <- ifelse(
      is.na(precision) | is.na(recall) | precision + recall == 0,
      NA,
      2 * precision * recall / (precision + recall)
    )
    
    data.frame(
      Class = label,
      Precision = precision,
      Recall = recall,
      F1 = f1
    )
  })
  
  data.frame(
    Model = model_name,
    Data_type = "all data",
    Total_observations = total_obs,
    Inflation_rate = round(inflation_rate * 100, 2),
    Accuracy = round(accuracy * 100, 2),
    Precision = round(mean(class_metrics$Precision, na.rm = TRUE) * 100, 2),
    Recall = round(mean(class_metrics$Recall, na.rm = TRUE) * 100, 2),
    F1_score = round(mean(class_metrics$F1, na.rm = TRUE) * 100, 2)
  )
}

llm_all_data_table <- bind_rows(
  evaluate_llm_all_data_inflation(results, "Agent Result", "Agent"),
  evaluate_llm_all_data_inflation(results, "GPT Result", "GPT"),
  evaluate_llm_all_data_inflation(results, "Gemini Result", "Gemini"),
  evaluate_llm_all_data_inflation(results, "LLAMA Result", "LLaMA"),
  evaluate_llm_all_data_inflation(results, "Claude Result", "Claude"),
  evaluate_llm_all_data_inflation(results, "Gemini Tuned Result", "Gemini Tuned")
  
)

llm_all_data_table
#--------------------BINARI-------------

final_labels <- c("book", "no book", "wrong", "misc")
binary_labels <- c("book", "not booked")

evaluate_llm_binary_all_data <- function(data, pred_col, model_name) {
  
  eval_data <- data %>%
    mutate(
      Truth_original = str_squish(str_to_lower(Result)),
      Prediction_original = str_squish(str_to_lower(.data[[pred_col]]))
    ) %>%
    filter(
      !is.na(Truth_original),
      Truth_original %in% final_labels
    ) %>%
    mutate(
      Truth_binary = if_else(
        Truth_original == "book",
        "book",
        "not booked"
      ),
      Prediction_binary = if_else(
        Prediction_original == "book",
        "book",
        "not booked"
      )
    )
  
  total_obs <- nrow(eval_data)
  
  accuracy <- mean(eval_data$Truth_binary == eval_data$Prediction_binary, na.rm = TRUE)
  
  class_metrics <- map_dfr(binary_labels, function(label) {
    
    TP <- sum(eval_data$Truth_binary == label & eval_data$Prediction_binary == label, na.rm = TRUE)
    FP <- sum(eval_data$Truth_binary != label & eval_data$Prediction_binary == label, na.rm = TRUE)
    FN <- sum(eval_data$Truth_binary == label & eval_data$Prediction_binary != label, na.rm = TRUE)
    
    precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
    recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
    f1 <- ifelse(
      is.na(precision) | is.na(recall) | precision + recall == 0,
      NA,
      2 * precision * recall / (precision + recall)
    )
    
    data.frame(
      Class = label,
      Precision = precision,
      Recall = recall,
      F1 = f1
    )
  })
  
  data.frame(
    Model = model_name,
    Data_type = "all data",
    Classification = "book vs not booked",
    Total_observations = total_obs,
    Accuracy = round(accuracy * 100, 2),
    Precision = round(mean(class_metrics$Precision, na.rm = TRUE) * 100, 2),
    Recall = round(mean(class_metrics$Recall, na.rm = TRUE) * 100, 2),
    F1_score = round(mean(class_metrics$F1, na.rm = TRUE) * 100, 2)
  )
}

llm_binary_all_data_table <- bind_rows(
  evaluate_llm_binary_all_data(results, "Agent Result", "Agent"),
  evaluate_llm_binary_all_data(results, "GPT Result", "GPT"),
  evaluate_llm_binary_all_data(results, "Gemini Result", "Gemini"),
  evaluate_llm_binary_all_data(results, "LLAMA Result", "LLaMA"),
  evaluate_llm_binary_all_data(results, "Claude Result", "Claude"),
  evaluate_llm_binary_all_data(results, "Gemini Tuned Result", "Gemini Tuned")
  
)

llm_binary_all_data_table

binary_check <- results %>%
  mutate(
    Ground_truth_binary = if_else(
      str_squish(str_to_lower(Result)) == "book",
      "book",
      "not booked"
    ),
    GPT_binary = if_else(
      str_squish(str_to_lower(`GPT Result`)) == "book",
      "book",
      "not booked"
    ),
    Gemini_binary = if_else(
      str_squish(str_to_lower(`Gemini Result`)) == "book",
      "book",
      "not booked"
    ),
    LLAMA_binary = if_else(
      str_squish(str_to_lower(`LLAMA Result`)) == "book",
      "book",
      "not booked"
    ),
    Gemini_Tuned_binary = if_else(
      str_squish(str_to_lower(`Gemini Tuned Result`)) == "book",
      "book",
      "not booked"
    ),
    Claude_binary = if_else(
      str_squish(str_to_lower(`Claude Result`)) == "book",
      "book",
      "not booked"
    )
  ) %>%
  select(
    Result,
    Ground_truth_binary,
    `GPT Result`,
    GPT_binary,
    `Gemini Result`,
    Gemini_binary,
    `LLAMA Result`,
    LLAMA_binary,
    `Gemini Tuned Result`,
    Gemini_Tuned_binary,
    `Claude Result`,
    Claude_binary
  )

head(binary_check, 20)
##-------ULTIMA------

final_labels <- c("book", "no book", "wrong", "misc")

evaluate_llm_by_class <- function(data, pred_col, model_name) {
  
  eval_data <- data %>%
    mutate(
      Truth = str_squish(str_to_lower(Result)),
      Prediction = str_squish(str_to_lower(.data[[pred_col]]))
    ) %>%
    filter(
      !is.na(Truth),
      Truth %in% final_labels
    )
  
  map_dfr(final_labels, function(label) {
    
    TP <- sum(eval_data$Truth == label & eval_data$Prediction == label, na.rm = TRUE)
    FP <- sum(eval_data$Truth != label & eval_data$Prediction == label, na.rm = TRUE)
    FN <- sum(eval_data$Truth == label & eval_data$Prediction != label, na.rm = TRUE)
    
    precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
    recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
    f1 <- ifelse(
      is.na(precision) | is.na(recall) | precision + recall == 0,
      NA,
      2 * precision * recall / (precision + recall)
    )
    
    data.frame(
      Model = model_name,
      Class = label,
      Precision = round(precision * 100, 2),
      Recall = round(recall * 100, 2),
      F1_score = round(f1 * 100, 2)
    )
  })
}

llm_class_metrics_long <- bind_rows(
  evaluate_llm_by_class(results, "GPT Result", "GPT"),
  evaluate_llm_by_class(results, "Gemini Result", "Gemini"),
  evaluate_llm_by_class(results, "LLAMA Result", "LLaMA"),
  evaluate_llm_by_class(results, "Claude Result", "Claude"),
  evaluate_llm_by_class(results, "Gemini Tuned Result", "Gemini Tuned")
  
)

llm_class_metrics_long
