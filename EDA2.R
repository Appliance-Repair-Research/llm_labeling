# ============================================================
# TIME SERIES EDA FOR CALL CENTER DATA
# Dataset columns: Time, Result, Duration
# File name: call_sheet_cleaned.xlsx
# ============================================================


# ------------------------------------------------------------
# 1. Load required libraries
# ------------------------------------------------------------

library(readxl)
library(dplyr)
library(ggplot2)
library(lubridate)
library(scales)
library(slider)
library(forecast)

setwd("~/Desktop/TFG/EDA")
# ------------------------------------------------------------
# 2. Load the Excel file
# ------------------------------------------------------------
calls <- read_csv(
  "call_sheet_cleaned.csv",
  show_col_types = FALSE
)

results=calls

category_distribution <- results %>%
  mutate(
    Result_clean = str_squish(str_to_lower(Result)),
    Result_clean = case_when(
      Result_clean == "rest" ~ "missed",
      TRUE ~ Result_clean
    )
  ) %>%
  count(Result_clean, name = "Number_of_calls") %>%
  mutate(
    Percentage = Number_of_calls / sum(Number_of_calls)
  ) %>%
  arrange(desc(Percentage))

category_distribution

ggplot(
  category_distribution,
  aes(x = reorder(Result_clean, Percentage), y = Percentage)
) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = percent(Percentage, accuracy = 0.1)),
    hjust = -0.1,
    size = 4
  ) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, max(category_distribution$Percentage) * 1.15)
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

# ------------------------------------------------------------
# 3. Inspect the structure of the dataset
# ------------------------------------------------------------

names(calls)
str(calls)
head(calls)
summary(calls)


# ------------------------------------------------------------
# 4. Convert Time into Date format
# Time originally looks like: 2022-01-03 00:00:00
# Since the hour is always 00:00:00, it can be treated as a date
# ------------------------------------------------------------
calls=calls[,-4]

calls <- calls %>%
  mutate(
    Time = as.Date(Time),
    Result = as.factor(Result),
    Duration = as.numeric(Duration)
  )


# ------------------------------------------------------------
# 5. Check missing values
# ------------------------------------------------------------

missing_values <- calls %>%
  summarise(
    missing_time = sum(is.na(Time)),
    missing_result = sum(is.na(Result)),
    missing_duration = sum(is.na(Duration))
  )

missing_values


# ------------------------------------------------------------
# 6. General dataset summary
# ------------------------------------------------------------

dataset_summary <- calls %>%
  summarise(
    total_calls = n(),
    first_day = min(Time, na.rm = TRUE),
    last_day = max(Time, na.rm = TRUE),
    number_of_days = n_distinct(Time),
    number_of_results = n_distinct(Result),
    mean_duration = mean(Duration, na.rm = TRUE),
    median_duration = median(Duration, na.rm = TRUE),
    min_duration = min(Duration, na.rm = TRUE),
    max_duration = max(Duration, na.rm = TRUE)
  )

dataset_summary


# ------------------------------------------------------------
# 7. Distribution of call results
# This is useful to detect class imbalance
# ------------------------------------------------------------

result_distribution <- calls %>%
  count(Result, sort = TRUE) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 2)
  )

result_distribution


ggplot(result_distribution, aes(x = reorder(Result, n), y = n, fill = Result)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Distribution of call results",
    x = "Call result",
    y = "Number of calls",
    fill = "Result"
  ) +
  theme_minimal()


# ------------------------------------------------------------
# 8. Define meaningful colors for call result categories
# Green is used for book, red/orange tones for negative outcomes,
# and grey tones for neutral or residual categories
# ------------------------------------------------------------

result_colors <- c(
  "book"    = "#2E7D32",  # green
  "no book" = "#C62828",  # red
  "missed"  = "#EF6C00",  # orange
  "wrong"   = "#D84315",  # red-orange
  "recall"  = "#757575",  # grey
  "rest"    = "#BDBDBD"   # light grey
)


# ------------------------------------------------------------
# 9. Order result categories for clearer visual interpretation
# ------------------------------------------------------------

calls <- calls %>%
  mutate(
    Result = factor(
      Result,
      levels = c("book", "no book", "missed", "wrong", "recall", "rest")
    )
  )


# ------------------------------------------------------------
# 10. Daily number of calls
# This creates a daily time series with total call volume
# ------------------------------------------------------------

daily_calls <- calls %>%
  group_by(Time) %>%
  summarise(
    total_calls = n(),
    .groups = "drop"
  ) %>%
  arrange(Time)

daily_calls


ggplot(daily_calls, aes(x = Time, y = total_calls)) +
  geom_line(linewidth = 0.4) +
  labs(
    title = "Daily call volume",
    x = "Date",
    y = "Number of calls"
  ) +
  theme_minimal()


# ------------------------------------------------------------
# 11. Daily call volume with moving averages
# Moving averages help smooth daily noise and reveal trends
# ------------------------------------------------------------

daily_calls <- daily_calls %>%
  mutate(
    ma_7 = slide_dbl(total_calls, mean, .before = 6, .complete = TRUE),
    ma_30 = slide_dbl(total_calls, mean, .before = 29, .complete = TRUE)
  )


ggplot(daily_calls, aes(x = Time)) +
  geom_line(aes(y = total_calls), alpha = 0.35) +
  geom_line(aes(y = ma_7), linewidth = 0.8) +
  geom_line(aes(y = ma_30), linewidth = 0.8) +
  labs(
    title = "Daily call volume with moving averages",
    x = "Date",
    y = "Number of calls"
  ) +
  theme_minimal()


# ------------------------------------------------------------
# 12. Weekly number of calls
# Weekly aggregation reduces noise and is easier to interpret
# ------------------------------------------------------------

weekly_calls <- calls %>%
  mutate(Week = floor_date(Time, "week", week_start = 1)) %>%
  group_by(Week) %>%
  summarise(
    total_calls = n(),
    .groups = "drop"
  ) %>%
  arrange(Week)

weekly_calls


ggplot(weekly_calls, aes(x = Week, y = total_calls)) +
  geom_line(linewidth = 0.6) +
  scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
  labs(
    title = "Weekly call volume",
    x = "Week",
    y = "Number of calls"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# ------------------------------------------------------------
# 14. Monthly call results in absolute values
# This shows how each result category evolves over time
# ------------------------------------------------------------

calls_monthly_result <- calls %>%
  mutate(Month = floor_date(Time, "month")) %>%
  count(Month, Result)

calls_monthly_result


ggplot(calls_monthly_result, aes(x = Month, y = n, fill = Result)) +
  geom_area(alpha = 0.9, position = "stack") +
  scale_fill_manual(values = result_colors) +
  scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
  labs(
    title = "Monthly evolution of call results",
    x = "Month",
    y = "Number of calls",
    fill = "Result"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# ------------------------------------------------------------
# 15. Monthly call results in percentages
# This is useful to compare the composition of results over time
# independently from the total number of calls
# ------------------------------------------------------------

calls_monthly_result_pct <- calls %>%
  mutate(Month = floor_date(Time, "month")) %>%
  count(Month, Result) %>%
  group_by(Month) %>%
  mutate(
    percentage = n / sum(n)
  ) %>%
  ungroup()

calls_monthly_result_pct


ggplot(calls_monthly_result_pct, aes(x = Month, y = percentage, fill = Result)) +
  geom_area(alpha = 0.9, position = "stack") +
  scale_fill_manual(values = result_colors) +
  scale_y_continuous(labels = percent_format()) +
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



# ------------------------------------------------------------
# 17. Weekly call volume by result category
# Weekly aggregation is smoother and usually better for interpretation
# ------------------------------------------------------------

weekly_by_result <- calls %>%
  mutate(Week = floor_date(Time, "week", week_start = 1)) %>%
  count(Week, Result)

weekly_by_result


ggplot(weekly_by_result, aes(x = Week, y = n, color = Result)) +
  geom_line(linewidth = 0.6) +
  scale_color_manual(values = result_colors) +
  facet_wrap(~ Result, scales = "free_y") +
  scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
  labs(
    title = "Weekly call volume by result category",
    x = "Week",
    y = "Number of calls",
    color = "Result"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# ------------------------------------------------------------
# 18. Weekly percentage of each result category
# This shows whether the proportion of each result changes over time
# ------------------------------------------------------------

weekly_result_pct <- calls %>%
  mutate(Week = floor_date(Time, "week", week_start = 1)) %>%
  count(Week, Result) %>%
  group_by(Week) %>%
  mutate(
    percentage = n / sum(n)
  ) %>%
  ungroup()

weekly_result_pct


ggplot(weekly_result_pct, aes(x = Week, y = percentage, color = Result)) +
  geom_line(linewidth = 0.6) +
  scale_color_manual(values = result_colors) +
  scale_y_continuous(labels = percent_format()) +
  facet_wrap(~ Result, scales = "free_y") +
  scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
  labs(
    title = "Weekly percentage of each call result",
    x = "Week",
    y = "Percentage of weekly calls",
    color = "Result"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# ------------------------------------------------------------
# 20. Weekly booking rate
# Weekly booking rate is smoother and more stable than daily booking rate
# ------------------------------------------------------------

weekly_book_rate <- calls %>%
  mutate(Week = floor_date(Time, "week", week_start = 1)) %>%
  count(Week, Result) %>%
  group_by(Week) %>%
  mutate(
    total_week = sum(n)
  ) %>%
  ungroup() %>%
  filter(Result == "book") %>%
  mutate(
    book_rate = n / total_week
  )

weekly_book_rate


ggplot(weekly_book_rate, aes(x = Week, y = book_rate)) +
  geom_line(linewidth = 0.6) +
  geom_smooth(se = FALSE) +
  scale_y_continuous(labels = percent_format()) +
  scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
  labs(
    title = "Weekly booking rate",
    x = "Week",
    y = "Booking rate"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# ------------------------------------------------------------
# 21. Duration summary by result
# This checks whether call duration differs by final result
# ------------------------------------------------------------

duration_by_result <- calls %>%
  group_by(Result) %>%
  summarise(
    n = n(),
    mean_duration = mean(Duration, na.rm = TRUE),
    median_duration = median(Duration, na.rm = TRUE),
    sd_duration = sd(Duration, na.rm = TRUE),
    min_duration = min(Duration, na.rm = TRUE),
    max_duration = max(Duration, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(median_duration))

duration_by_result


ggplot(calls, aes(x = Result, y = Duration, fill = Result)) +
  geom_boxplot(outlier.alpha = 0.15) +
  scale_fill_manual(values = result_colors) +
  coord_flip() +
  labs(
    title = "Call duration by result",
    x = "Call result",
    y = "Duration"
  ) +
  theme_minimal()

# ------------------------------------------------------------
# 23. Call result by weekday
# This shows whether the result distribution changes depending on the weekday
# ------------------------------------------------------------

calls_weekday_result <- calls %>%
  count(Weekday, Result)

calls_weekday_result


ggplot(calls_weekday_result, aes(x = Weekday, y = n, fill = Result)) +
  geom_col() +
  scale_fill_manual(values = result_colors) +
  labs(
    title = "Call results by weekday",
    x = "Weekday",
    y = "Number of calls",
    fill = "Result"
  ) +
  theme_minimal()


# ------------------------------------------------------------
# 24. Create a complete daily time series
# This is important because time series models require regular time intervals
# Missing dates are filled with zero calls
# ------------------------------------------------------------

complete_daily_calls <- daily_calls %>%
  select(Time, total_calls) %>%
  right_join(
    tibble(
      Time = seq(
        from = min(daily_calls$Time, na.rm = TRUE),
        to = max(daily_calls$Time, na.rm = TRUE),
        by = "day"
      )
    ),
    by = "Time"
  ) %>%
  mutate(
    total_calls = ifelse(is.na(total_calls), 0, total_calls)
  ) %>%
  arrange(Time)

complete_daily_calls


# ------------------------------------------------------------
# 25. STL decomposition of daily call volume
# Frequency = 7 assumes weekly seasonality
# This decomposes the series into trend, seasonality, and remainder
# ------------------------------------------------------------

ts_calls <- ts(
  complete_daily_calls$total_calls,
  frequency = 7
)

decomp_calls <- stl(ts_calls, s.window = "periodic")

plot(decomp_calls)


# ------------------------------------------------------------
# 26. ARIMA model for daily call volume
# This forecasts total daily call volume
# It does not predict the result of each call
# ------------------------------------------------------------

fit_arima_calls <- auto.arima(ts_calls)

summary(fit_arima_calls)

# ------------------------------------------------------------
# 28. Optional weekly time series model
# Weekly aggregation can produce a smoother and more stable forecast
# ------------------------------------------------------------

ts_weekly_calls <- ts(
  weekly_calls$total_calls,
  frequency = 52
)

fit_arima_weekly <- auto.arima(ts_weekly_calls)

summary(fit_arima_weekly)


forecast_weekly_calls <- forecast(fit_arima_weekly, h = 8)

autoplot(forecast_weekly_calls) +
  labs(
    title = "Forecast of weekly call volume",
    x = "Time",
    y = "Number of calls"
  ) +
  theme_minimal()

# ------------------------------------------------------------
# Remove Sundays from the dataset
# The call center is not operational on Sundays, so Sundays are
# excluded from the daily time series model
# ------------------------------------------------------------

calls_no_sunday <- calls %>%
  mutate(
    Weekday = wday(Time, label = TRUE, week_start = 1)
  ) %>%
  filter(Weekday != "Sun")

# ------------------------------------------------------------
# Daily call volume excluding Sundays
# ------------------------------------------------------------

daily_calls_no_sunday <- calls_no_sunday %>%
  group_by(Time) %>%
  summarise(
    total_calls = n(),
    .groups = "drop"
  ) %>%
  arrange(Time)

ggplot(daily_calls_no_sunday, aes(x = Time, y = total_calls)) +
  geom_line(linewidth = 0.4) +
  labs(
    title = "Daily call volume excluding Sundays",
    x = "Date",
    y = "Number of calls"
  ) +
  theme_minimal()
# ------------------------------------------------------------
# ARIMA model for daily call volume excluding Sundays
# Frequency = 6 because the operational week has six working days
# ------------------------------------------------------------

ts_calls_no_sunday <- ts(
  daily_calls_no_sunday$total_calls,
  frequency = 6
)

fit_arima_calls_no_sunday <- auto.arima(ts_calls_no_sunday)

summary(fit_arima_calls_no_sunday)



# ------------------------------------------------------------
# 29. Main interpretation points for the report
# Use the outputs above to discuss:
# - Whether total call volume is stable or variable over time
# - Whether there is weekly seasonality
# - Whether some result categories grow or decline over time
# - Whether the booking rate is stable or changes across weeks/months
# - Whether some categories are too small or noisy to model separately
# - Whether call duration differs by result category
# - Whether time series forecasting is useful for operational planning
# ------------------------------------------------------------