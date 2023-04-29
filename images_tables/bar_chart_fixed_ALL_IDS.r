# Load required libraries
library(ggplot2)
library(tidyr)
library(dplyr)

# Read the CSV data
filename <- "images_tables_thesis/1_hate_speech/monolingual_setting/Incel BERT 1M ALL NON-incelsis IDs.csv"
data <- read.csv(filename) # nolint

# Sort the data by test_f1
data <- data %>% arrange(desc(test_f1))

# Reshape the data to long format
data_long <- gather(data, key = "metric", value = "value", c(val_f1, test_f1))
print(data_long)
data_std_long <- gather(data, key = "metric_std", value = "value_std", c(val_f1_std, test_f1_std)) # nolint
print(data_std_long)

# Merge the reshaped data
data_long$value_std <- data_std_long$value_std

# Create a helper function for sorting the x-axis
sort_by_test_f1 <- function(run_id) {
  order_factor <- match(run_id, data$run_id)
  data$test_f1[order_factor]
}

model <- data$model[1]

# Create the bar chart with error bars
bar_chart <- ggplot(
  data_long,
  aes(x = reorder(run_id, -sort_by_test_f1(run_id)),
    y = value,
    fill = metric)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(
    aes(ymin = value - value_std,
    ymax = value + value_std),
    width = 0.2,
    position = position_dodge(0.9)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        # Set the legend position
        legend.position = c(1, 1),
        # Set the legend anchor point
        legend.justification = c(1, 1)) +
  labs(
    # title = "Validation and Test F1 Scores with Error Bars",
  x = "Run ID",
  y = "F1 Score") +
  scale_fill_discrete(
    name = paste("Model: ", model, "\nMetric"),
    labels = c("Validation F1", "Test F1")) +
  coord_cartesian(ylim = c(0, 0.9))

# Save the plot as a file
print(bar_chart)
new_csv_filename <- paste0(substr(filename, 1, nchar(filename) - 4), "_R.png")
ggsave(new_csv_filename)