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

refrigerator <- read_csv("refrigerator.csv")
oven <- read_csv("oven.csv")
dishwasher <- read_csv("dishwasher.csv")
appliance <- read_csv("appliance.csv")

# ------------------------------------------------------------
# 2. Standardise column names and add keyword label
# ------------------------------------------------------------

refrigerator <- refrigerator %>%
  rename(Trend = `refrigerator repair`) %>%
  mutate(Keyword = "Refrigerator repair")

oven <- oven %>%
  rename(Trend = `oven repair`) %>%
  mutate(Keyword = "Oven repair")

dishwasher <- dishwasher %>%
  rename(Trend = `dishwasher repair`) %>%
  mutate(Keyword = "Dishwasher repair")

appliance <- appliance %>%
  rename(Trend = `appliance repair`) %>%
  mutate(Keyword = "Appliance repair")

# ------------------------------------------------------------
# 3. Combine all Google Trends datasets
# ------------------------------------------------------------

google_trends <- bind_rows(
  refrigerator,
  oven,
  dishwasher,
  appliance
) %>%
  mutate(
    Time = as.Date(Time),
    Year = year(Time),
    Month = month(Time),
    Month_name = month(Time, label = TRUE, abbr = FALSE)
  )

# ============================================================
# GOOGLE TRENDS 2004–2020 + DETRENDED APPLIANCE REPAIR
# ============================================================
google_trends_2004_2020 <- google_trends %>%
  filter(
    Time >= as.Date("2004-01-01"),
    Time <= as.Date("2020-12-31")
  )

# ------------------------------------------------------------
# Plot four separate time series together using facets
# ------------------------------------------------------------

ggplot(google_trends_2004_2020, aes(x = Time, y = Trend)) +
  geom_line(linewidth = 0.7) +
  facet_wrap(~ Keyword, ncol = 2) +
  scale_x_date(date_breaks = "4 years", date_labels = "%Y") +
  labs(
    title = "Google Trends interest over time by keyword",
    subtitle = "Period 2004–2020",
    x = "Year",
    y = "Google Trends index"
  ) +
  theme_minimal()

# ============================================================
# GOOGLE TRENDS — FOUR SEPARATE PLOTS TOGETHER
# Period: 2015
# ============================================================

google_trends_2015_2016 <- google_trends %>%
  filter(
    Time >= as.Date("2015-01-01"),
    Time <= as.Date("2016-12-31")
  )

ggplot(google_trends_2015_2016, aes(x = Time, y = Trend)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.6) +
  facet_wrap(~ Keyword, ncol = 2) +
  scale_x_date(
    date_breaks = "2 months",
    date_labels = "%b %Y"
  ) +
  labs(
    title = "Google Trends interest over time by keyword",
    subtitle = "Period 2015–2016",
    x = "Month",
    y = "Google Trends index"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

#-------
#Taula
google_trends_detrended <- google_trends_2004_2020 %>%
  group_by(Keyword) %>%
  arrange(Time, .by_group = TRUE) %>%
  mutate(
    time_index = row_number(),
    trend_loess = predict(
      loess(Trend ~ time_index, span = 0.35),
      newdata = data.frame(time_index = time_index)
    ),
    detrended_trend = Trend - trend_loess
  ) %>%
  ungroup()

monthly_influence_detrended <- google_trends_detrended %>%
  mutate(
    Month_num = month(Time),
    Month = month(Time, label = TRUE, abbr = FALSE)
  ) %>%
  group_by(Month_num, Month, Keyword) %>%
  summarise(
    average_detrended_influence = round(mean(detrended_trend, na.rm = TRUE), 2),
    .groups = "drop"
  ) %>%
  select(-Month_num) %>%
  pivot_wider(
    names_from = Keyword,
    values_from = average_detrended_influence
  ) %>%
  arrange(match(Month, month.name))

monthly_influence_detrended

# ------------------------------------------------------------
# Simple decomposition function for each Google Trends series
# ------------------------------------------------------------

decompose_google_trends <- function(data, keyword_name) {
  
  series_data <- data %>%
    filter(Keyword == keyword_name) %>%
    arrange(Time)
  
  trend_ts <- ts(
    series_data$Trend,
    start = c(year(min(series_data$Time)), month(min(series_data$Time))),
    frequency = 12
  )
  
  decomposition <- decompose(trend_ts)
  
  plot(decomposition)
  
  return(decomposition)
}

# ------------------------------------------------------------
# Apply decomposition to each Google Trends keyword
# ------------------------------------------------------------

refrigerator_decomposition <- decompose_google_trends(
  google_trends_2004_2020,
  "Refrigerator repair"
)

oven_decomposition <- decompose_google_trends(
  google_trends_2004_2020,
  "Oven repair"
)

dishwasher_decomposition <- decompose_google_trends(
  google_trends_2004_2020,
  "Dishwasher repair"
)

appliance_decomposition <- decompose_google_trends(
  google_trends_2004_2020,
  "Appliance repair"
)

# ------------------------------------------------------------
# Save decomposition plots
# ------------------------------------------------------------

png("decomposition_refrigerator_repair.png", width = 1000, height = 700)
plot(refrigerator_decomposition)
dev.off()

png("decomposition_oven_repair.png", width = 1000, height = 700)
plot(oven_decomposition)
dev.off()

png("decomposition_dishwasher_repair.png", width = 1000, height = 700)
plot(dishwasher_decomposition)
dev.off()

png("decomposition_appliance_repair.png", width = 1000, height = 700)
plot(appliance_decomposition)
dev.off()
