
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

library(readr)
library(dplyr)
library(lubridate)

calls <- read_csv(
  "call_sheet_cleaned.csv",
  show_col_types = FALSE
)

calls <- calls %>%
  mutate(
    Time = mdy(Time)
  )

last_date <- max(calls$Time, na.rm = TRUE)

daily_calls_last_6_months_no_sundays <- calls %>%
  filter(
    Time >= last_date %m-% months(6),
    wday(Time, label = TRUE, week_start = 1) != "Sun"
  ) %>%
  count(Time, name = "Number_of_calls")

mean_daily_calls_last_6_months_no_sundays <- daily_calls_last_6_months_no_sundays %>%
  summarise(
    mean_daily_calls = mean(Number_of_calls, na.rm = TRUE)
  )

mean_daily_calls_last_6_months_no_sundays

# ---------------

calls <- calls %>%
  mutate(
    `Baku Time` = format(
      with_tz(
        dmy_hm(`Baku Time`, tz = "Europe/Madrid"),
        tzone = "America/New_York"
      ),
      "%H:%M"
    )
  )

calls <- calls %>%
  mutate(
    Time_minutes = as.numeric(substr(`Baku Time`, 1, 2)) * 60 +
      as.numeric(substr(`Baku Time`, 4, 5)),
    
    Time_slot = case_when(
      Time_minutes >= 8 * 60  & Time_minutes < 12 * 60 ~ "Morning",
      Time_minutes >= 12 * 60 & Time_minutes < 15 * 60 ~ "Midday",
      Time_minutes >= 15 * 60 & Time_minutes < 18 * 60 ~ "Afternoon"
    )
  ) %>%
  select(-Time_minutes)


calls_weekday <- calls %>%
  mutate(
    Date = mdy(as.character(Time)),
    Weekday = wday(
      Date,
      label = TRUE,
      abbr = FALSE,
      week_start = 1
    )
  ) %>%
  filter(!is.na(Date), !is.na(Weekday)) %>%
  count(Weekday, name = "Number_of_calls") %>%
  mutate(
    Percentage = Number_of_calls / sum(Number_of_calls)
  )

# Veure la taula
calls_weekday

# Gràfic de barres
ggplot(
  calls_weekday,
  aes(x = Weekday, y = Percentage)
) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = percent(Percentage, accuracy = 0.1)),
    vjust = -0.4,
    size = 4
  ) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, max(calls_weekday$Percentage) * 1.15)
  ) +
  labs(
    title = "Distribution of calls by day of the week",
    x = "Day of the week",
    y = "Percentage of calls"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )

# ------------------------------------------------------------
# 1. Check number of calls by weekday
# This allows us to verify that Sundays are not regular working days
# Time is parsed only temporarily; the original Time column is not changed
# ------------------------------------------------------------

calls_weekday_check <- calls %>%
  mutate(
    Date_plot = mdy(Time),
    Weekday = wday(Date_plot, label = TRUE, week_start = 1)
  ) %>%
  count(Weekday) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 2)
  )

calls_weekday_check


# ------------------------------------------------------------
# 2. Daily number of calls including all days
# This uses Date_plot only for plotting and aggregation
# ------------------------------------------------------------

daily_calls_all <- calls %>%
  mutate(
    Date_plot = mdy(Time)
  ) %>%
  group_by(Date_plot) %>%
  summarise(
    daily_calls = n(),
    .groups = "drop"
  ) %>%
  arrange(Date_plot)

ggplot(daily_calls_all, aes(x = Date_plot, y = daily_calls, group = 1)) +
  geom_line(linewidth = 0.4) +
  scale_x_date(date_breaks = "3 months", date_labels = "%m/%d/%y") +
  labs(
    title = "Daily number of calls received",
    x = "Date",
    y = "Number of calls"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# ------------------------------------------------------------
# 3. Remove Sundays for operational analysis
# The original Time column is still not modified
# ------------------------------------------------------------

calls_no_sunday <- calls %>%
  mutate(
    Date_plot = mdy(Time),
    Weekday = wday(Date_plot, label = TRUE, week_start = 1)
  ) %>%
  filter(Weekday != "Sun")


# ------------------------------------------------------------
# 4. Daily number of calls excluding Sundays
# This is the preferred daily plot because the call center does not work on Sundays
# ------------------------------------------------------------

daily_calls_no_sunday <- calls_no_sunday %>%
  group_by(Date_plot) %>%
  summarise(
    daily_calls = n(),
    .groups = "drop"
  ) %>%
  arrange(Date_plot)

ggplot(daily_calls_no_sunday, aes(x = Date_plot, y = daily_calls, group = 1)) +
  geom_line(linewidth = 0.4) +
  scale_x_date(date_breaks = "3 months", date_labels = "%m/%d/%y") +
  labs(
    title = "Daily number of calls received excluding Sundays",
    x = "Date",
    y = "Number of calls"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# ------------------------------------------------------------
# 5. Monthly number of calls excluding Sundays
# This gives a clearer view of the long-term evolution
# ------------------------------------------------------------

monthly_calls <- calls_no_sunday %>%
  mutate(
    Month = floor_date(Date_plot, "month")
  ) %>%
  group_by(Month) %>%
  summarise(
    monthly_calls = n(),
    .groups = "drop"
  ) %>%
  arrange(Month)

monthly_calls

ggplot(monthly_calls, aes(x = Month, y = monthly_calls)) +
  geom_col() +
  scale_x_date(date_breaks = "3 months", date_labels = "%m/%y") +
  labs(
    title = "Monthly number of calls received",
    x = "Month",
    y = "Number of calls"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# ------------------------------------------------------------
# 6. Summary statistics for daily call volume excluding Sundays
# ------------------------------------------------------------

daily_calls_summary <- daily_calls_no_sunday %>%
  summarise(
    total_operational_days = n(),
    total_calls = sum(daily_calls),
    mean_daily_calls = mean(daily_calls),
    median_daily_calls = median(daily_calls),
    min_daily_calls = min(daily_calls),
    max_daily_calls = max(daily_calls),
    sd_daily_calls = sd(daily_calls)
  )

daily_calls_summary


# ------------------------------------------------------------
# 7. Summary statistics for monthly call volume excluding Sundays
# ------------------------------------------------------------

monthly_calls_summary <- monthly_calls %>%
  summarise(
    total_months = n(),
    total_calls = sum(monthly_calls),
    mean_monthly_calls = mean(monthly_calls),
    median_monthly_calls = median(monthly_calls),
    min_monthly_calls = min(monthly_calls),
    max_monthly_calls = max(monthly_calls),
    sd_monthly_calls = sd(monthly_calls)
  )

monthly_calls_summary


#------

calls %>%
  count(Result, sort = TRUE) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 2)
  )

# ------------------------------------------------------------
# 1. Prepare dates without modifying the original Time column
# ------------------------------------------------------------

calls_complete_years <- calls %>%
  mutate(
    Date_plot = mdy(Time),
    Year = year(Date_plot),
    Month = month(Date_plot, label = TRUE, abbr = FALSE),
    Month_num = month(Date_plot),
    Weekday = wday(Date_plot, label = TRUE, week_start = 1)
  ) %>%
  filter(
    Weekday != "Sun",
    Year >= 2022,
    Year <= 2025
  )

# ------------------------------------------------------------
# 2. Monthly call volume by year
# This gives total calls for each month in each complete year
# ------------------------------------------------------------

monthly_by_year <- calls_complete_years %>%
  group_by(Year, Month_num, Month) %>%
  summarise(
    monthly_calls = n(),
    .groups = "drop"
  ) %>%
  arrange(Year, Month_num)

monthly_by_year

# ------------------------------------------------------------
# 3. Average call volume by month of the year
# This gives values such as average calls in January, February, etc.
# across complete years
# ------------------------------------------------------------

monthly_average <- monthly_by_year %>%
  group_by(Month_num, Month) %>%
  summarise(
    average_monthly_calls = mean(monthly_calls),
    median_monthly_calls = median(monthly_calls),
    min_monthly_calls = min(monthly_calls),
    max_monthly_calls = max(monthly_calls),
    .groups = "drop"
  ) %>%
  arrange(desc(average_monthly_calls))

monthly_average

