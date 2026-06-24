library(readxl)
library(dplyr)
library(ggplot2)
library(lubridate)
library(janitor)

setwd("~/Desktop/TFG/EDA")

calls <- read_csv(
  "call_sheet_cleaned.csv",
  show_col_types = FALSE
)

calls=calls[,-4]
calls <- calls %>%
  mutate(
    time = as.Date(Time),
    result = as.factor(Result),
    duration = as.numeric(Duration)
  )

#A. Distribució dels resultats de trucada

result_distribution <- calls %>%
  count(Result, sort = TRUE) %>%
  mutate(percentage = n / sum(n) * 100)

result_distribution

ggplot(result_distribution, aes(x = reorder(Result, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Distribution of call results",
    x = "Call result",
    y = "Number of calls"
  ) +
  theme_minimal()

# B. Nombre de trucades per dia
calls_monthly <- calls %>%
  mutate(Month = floor_date(Time, "month")) %>%
  count(Month)

ggplot(calls_monthly, aes(x = Month, y = n)) +
  geom_col() +
  labs(
    title = "Number of calls per month",
    x = "Month",
    y = "Number of calls"
  ) +
  theme_minimal()

# 7. Resultats al llarg del temps

## Valors absoluts

calls %>%
  count(Time, Result) %>%
  ggplot(aes(x = Time, y = n, fill = Result)) +
  geom_col() +
  labs(
    title = "Call results over time",
    x = "Date",
    y = "Number of calls",
    fill = "Result"
  ) +
  theme_minimal()

## Percentatges
result_colors <- c(
  "book"    = "#2E7D32",  # green
  "no book" = "#C62828",  # red
  "missed"  = "#EF6C00",  # orange
  "wrong"   = "#D84315",  # red-orange
  "recall"  = "#757575",  # grey
  "rest"    = "#BDBDBD"   # light grey
)

calls_monthly_result_pct <- calls %>%
  mutate(Month = floor_date(Time, "month")) %>%
  count(Month, Result) %>%
  group_by(Month) %>%
  mutate(percentage = n / sum(n) * 100) %>%
  ungroup()

  
  ggplot(calls_monthly_result_pct, aes(x = Month, y = percentage, fill = Result)) +
  geom_area(alpha = 0.85, position = "stack") +
  scale_fill_manual(values = result_colors) +
  scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
  labs(
    title = "Monthly percentage distribution of call results",
    x = "Month",
    y = "Percentage of calls",
    fill = "Result"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
  