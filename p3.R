library(readr)
library(dplyr)
library(ggplot2)
library(scales)

setwd("~/Desktop/TFG/EDA")

calls <- read_csv(
  "call_sheet_cleaned.csv",
  show_col_types = FALSE
)

calls <- calls %>%
  mutate(
    Result = if_else(Result == "rest", "misc", Result)
  )

total_result_distribution <- calls %>%
  count(Result, sort = TRUE) %>%
  mutate(
    percentage = n / sum(n)
  )

ggplot(
  total_result_distribution,
  aes(x = reorder(Result, percentage), y = percentage)
) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = percent(percentage, accuracy = 0.1)),
    hjust = -0.1,
    size = 4
  ) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, max(total_result_distribution$percentage) * 1.15)
  ) +
  coord_flip() +
  labs(
    title = "Distribution of calls by final category",
    x = "Category",
    y = "Percentage of calls"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  filename = "total_result_distribution.png",
  width = 8,
  height = 5,
  dpi = 300
)

result_2026_table <- calls %>%
  mutate(
    Date_plot = mdy(Time),
    Year = year(Date_plot)
  ) %>%
  filter(Year == 2026) %>%
  count(Result, sort = TRUE) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 2)
  )

result_2026_table
