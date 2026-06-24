# ============================================================
# DURATION DATA AVAILABILITY
# Check from when to when Duration data is available
# ============================================================

library(dplyr)
library(lubridate)

setwd("~/Desktop/TFG/EDA")
# ------------------------------------------------------------
# 2. Load the Excel file
# ------------------------------------------------------------
calls <- read_csv(
  "call_sheet_cleaned.csv",
  show_col_types = FALSE
)



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

duration_availability <- calls %>%
  mutate(
    Date_plot = mdy(Time)
  ) %>%
  filter(
    !is.na(Duration)
  ) %>%
  summarise(
    first_duration_date = min(Date_plot, na.rm = TRUE),
    last_duration_date = max(Date_plot, na.rm = TRUE),
    records_with_duration = n()
  )

duration_availability

# ============================================================
# BOXPLOT OF CALL DURATION BY MONTH WITHOUT EXTREME OUTLIERS
# ============================================================

library(dplyr)
library(lubridate)
library(ggplot2)
library(readr)

duration_monthly <- calls %>%
  mutate(
    Date_plot = mdy(Time),
    Month = floor_date(Date_plot, "month"),
    Duration_num = parse_number(as.character(Duration))
  ) %>%
  filter(
    !is.na(Duration_num)
  )

duration_p95 <- quantile(duration_monthly$Duration_num, 0.95, na.rm = TRUE)

ggplot(duration_monthly, aes(x = Month, y = Duration_num, group = Month)) +
  geom_boxplot(outlier.shape = NA) +
  coord_cartesian(ylim = c(0, duration_p95)) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %Y") +
  labs(
    title = "Distribution of call duration by month",
    subtitle = "Outliers hidden and y-axis limited to the 95th percentile",
    x = "Month",
    y = "Call duration"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 60, hjust = 1)
  )
