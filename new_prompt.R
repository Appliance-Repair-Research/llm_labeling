library(dplyr)
library(stringr)
library(purrr)
library(ggplot2)

setwd("~/Desktop/TFG/CODE")

results<- read_csv("call_data_revised.csv")

final_labels <- c("book", "no book", "wrong", "misc")

# Revisa que els noms coincideixin amb les columnes del teu dataset
names(results)

new_prompt_models <- c(
  "GPT New Prompt" = "GPT Result",
  "Gemini New Prompt" = "Gemini Result",
  "LLaMA New Prompt" = "LLAMA Result",
  "Claude New Prompt" = "Claude Result"
  
)

evaluate_llm_new_prompt <- function(data, pred_col, model_name) {
  
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
    Prompt = "revised prompt",
    Total_observations = total_obs,
    Inflation_rate = round(inflation_rate * 100, 2),
    Accuracy = round(accuracy * 100, 2),
    Precision = round(mean(class_metrics$Precision, na.rm = TRUE) * 100, 2),
    Recall = round(mean(class_metrics$Recall, na.rm = TRUE) * 100, 2),
    F1_score = round(mean(class_metrics$F1, na.rm = TRUE) * 100, 2)
  )
}

llm_new_prompt_table <- imap_dfr(
  new_prompt_models,
  ~ evaluate_llm_new_prompt(
    data = results,
    pred_col = .x,
    model_name = .y
  )
)

llm_new_prompt_table
