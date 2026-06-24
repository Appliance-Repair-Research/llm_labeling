library(readr)
library(dplyr)
library(ggplot2)
library(lubridate)
library(stringr)

# ------------------------------------------------------------
# Function to read Google Trends file using only column 2
# ------------------------------------------------------------

read_google_trends <- function(file_name, keyword_name) {
  
  data <- read_csv(
    file_name,
    skip = 3,
    col_names = FALSE,
    col_types = cols(.default = col_character()),
    show_col_types = FALSE
  )
  
  data_clean <- data %>%
    select(Interest = 2) %>%
    mutate(
      Interest = str_replace(Interest, "<1", "0"),
      Interest = as.numeric(Interest)
    ) %>%
    filter(!is.na(Interest)) %>%
    mutate(
      Date = seq(
        from = ym("2004-01"),
        by = "1 month",
        length.out = n()
      ),
      Keyword = keyword_name
    ) %>%
    select(Date, Interest, Keyword)
  
  return(data_clean)
}

# ------------------------------------------------------------
# Read each Google Trends CSV
# ------------------------------------------------------------

appliance_clean <- read_google_trends(
  "appliance.csv",
  "Appliance repair"
)

dishwasher_clean <- read_google_trends(
  "dishwasher.csv",
  "Dishwasher repair"
)

oven_clean <- read_google_trends(
  "oven.csv",
  "Oven repair"
)

refrigerator_clean <- read_google_trends(
  "refrigerator.csv",
  "Refrigerator repair"
)

# ------------------------------------------------------------
# Combine all datasets
# ------------------------------------------------------------

trends_all <- bind_rows(
  appliance_clean,
  dishwasher_clean,
  oven_clean,
  refrigerator_clean
)

# ------------------------------------------------------------
# Plot four graphs in one figure
# ------------------------------------------------------------

google_trends_faceted_plot <- ggplot(
  trends_all,
  aes(x = Date, y = Interest)
) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ Keyword, ncol = 2, scales = "free_y") +
  labs(
    title = "Google Trends interest over time by keyword",
    subtitle = "Period 2004–2026",
    x = "Year",
    y = "Google Trends interest"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0),
    plot.subtitle = element_text(hjust = 0),
    strip.text = element_text(size = 11),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

google_trends_faceted_plot

# ------------------------------------------------------------
# Save plot for Overleaf
# ------------------------------------------------------------

ggsave(
  filename = "google_trends_four_keywords.png",
  plot = google_trends_faceted_plot,
  width = 9,
  height = 6,
  dpi = 300
)
