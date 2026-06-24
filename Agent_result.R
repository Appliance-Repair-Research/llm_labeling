library(dplyr)
library(stringr)
library(ggplot2)

setwd("~/Desktop/TFG/CODE")

results<- read_csv("call_data_rw.csv")


final_labels <- c("book", "no book", "wrong", "misc")

agent_eval <- results %>%
  mutate(
    Truth = str_squish(str_to_lower(Result)),
    Agent_raw = str_squish(str_to_lower(`Agent Result`))
  ) %>%
  filter(
    !is.na(Truth),
    Truth %in% final_labels
  ) %>%
  mutate(
    Agent_valid = if_else(
      Agent_raw %in% final_labels,
      Agent_raw,
      "invalid"
    )
  )

agent_raw_output_distribution <- agent_eval %>%
  count(Agent_raw, sort = TRUE) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 2)
  )

agent_raw_output_distribution

agent_inflation_ratio <- agent_eval %>%
  summarise(
    total_predictions = n(),
    invalid_predictions = sum(Agent_valid == "invalid"),
    inflation_ratio = invalid_predictions / total_predictions,
    inflation_ratio_percentage = round(inflation_ratio * 100, 2)
  )

agent_inflation_ratio

agent_eval_valid <- agent_eval %>%
  filter(Agent_valid %in% final_labels) %>%
  mutate(
    Truth = factor(Truth, levels = final_labels),
    Agent_valid = factor(Agent_valid, levels = final_labels)
  )

agent_confusion_matrix <- table(
  Truth = agent_eval_valid$Truth,
  Prediction = agent_eval_valid$Agent_valid
)

agent_confusion_matrix

agent_accuracy <- sum(diag(agent_confusion_matrix)) / sum(agent_confusion_matrix)

agent_accuracy

agent_class_metrics <- lapply(final_labels, function(label) {
  
  TP <- agent_confusion_matrix[label, label]
  FP <- sum(agent_confusion_matrix[, label]) - TP
  FN <- sum(agent_confusion_matrix[label, ]) - TP
  
  precision <- ifelse(TP + FP == 0, NA, TP / (TP + FP))
  recall <- ifelse(TP + FN == 0, NA, TP / (TP + FN))
  f1_score <- ifelse(
    is.na(precision) | is.na(recall) | precision + recall == 0,
    NA,
    2 * precision * recall / (precision + recall)
  )
  
  data.frame(
    Class = label,
    Support = sum(agent_confusion_matrix[label, ]),
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

agent_class_metrics

agent_validation_summary <- data.frame(
  Model = "Agent Result",
  Total_cases = nrow(agent_eval),
  Valid_predictions = nrow(agent_eval_valid),
  Invalid_predictions = agent_inflation_ratio$invalid_predictions,
  Inflation_ratio = agent_inflation_ratio$inflation_ratio_percentage,
  Accuracy = round(agent_accuracy * 100, 2),
  Macro_precision = round(mean(agent_class_metrics$Precision, na.rm = TRUE), 2),
  Macro_recall = round(mean(agent_class_metrics$Recall, na.rm = TRUE), 2),
  Macro_F1 = round(mean(agent_class_metrics$F1_score, na.rm = TRUE), 2)
)

agent_validation_summary

agent_confusion_df <- as.data.frame(agent_confusion_matrix)

ggplot(agent_confusion_df, aes(x = Prediction, y = Truth, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 5) +
  labs(
    title = "Agent Result Confusion Matrix",
    x = "Agent result",
    y = "Ground truth"
  ) +
  theme_minimal()