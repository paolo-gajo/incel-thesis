library(ggplot2)
library(tidyr)
library(dplyr)

# Read the CSV data
filename <- "images_tables_thesis/1_hate_speech/monolingual_setting/All Models - ID 0.csv"
data <- read.csv(filename) # nolint

# Sort the data by test_f1
data <- data %>% arrange(desc(test_f1))

# Reshape the data to long format
data_long <- gather(data, key = "metric", value = "value", c(val_f1, test_f1))
data_std_long <- gather(data, key = "metric_std", value = "value_std", c(val_f1_std, test_f1_std)) # nolint

# Merge the reshaped data
data_long$value_std <- data_std_long$value_std

# Create a helper function for sorting the x-axis
sort_by_test_f1 <- function(model) {
  order_factor <- match(model, data$model)
  data$test_f1[order_factor]
}

run_id <- 0

# Define plot height and bar width
plot_height <- 4 # Modify this value to change the plot height
bar_width <- plot_height / (0.5 * nrow(data))

# Create the bar chart with error bars
bar_chart <- ggplot(
  data_long,
  aes(x = reorder(model, -sort_by_test_f1(model)),
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
    x = "Model",
    y = "F1 Score") +
  scale_fill_discrete(
    name = paste("Run ID: ", run_id),
    labels = c("Test F1", "Val F1")) +
  coord_cartesian(ylim = c(0.75, 0.9))


# Save the plot as a file
print(bar_chart)
new_csv_filename <- paste0(substr(filename, 1, nchar(filename) - 4), "_R.png")
ggsave(new_csv_filename)
