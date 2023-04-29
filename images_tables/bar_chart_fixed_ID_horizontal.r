library(ggplot2)
library(tidyr)
library(dplyr)
library(grid)  # Add this line

# Read the CSV data
filename <- "images_tables_thesis/2_multi-label/racist/multilingual_setting/ID_17_all_models.csv"
# filename <- "images_tables_thesis/1_hate_speech/monolingual_setting/Best and Baseline - ID -1 (random init).csv"
data <- read.csv(filename) # nolint

# Sort the data by test_f1
data <- data %>% arrange(desc(test_f1))
print(data)
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

run_id <- 17

# Define fixed bar width and calculate plot height
bar_width <- 0.9
plot_height <- nrow(data)

if plot_height < 4 { # nolint: error.
  plot_height <- 4
}

# Add epoch values to the data_long dataframe
data_long$epoch <- rep(data$epoch)

x_left_lim <- 0
x_right_lim <- 1

# Create the horizontal bar chart
bar_chart <- ggplot(
  data_long,
  aes(y = reorder(model, sort_by_test_f1(model)),
      x = value,
      fill = metric)) +
  geom_bar(stat = "identity", position = position_dodge(), width = bar_width) +
  geom_errorbarh(
    aes(xmin = value - value_std,
        xmax = value + value_std),
    height = bar_width * 0.8,
    position = position_dodge(bar_width * 1.1)) +
  geom_text(data = data_long %>% filter(metric == "test_f1"),
            aes(label = paste0('Epoch: ', epoch), x = x_left_lim),
            hjust = 0, vjust = 0.5,
            position = position_dodge(bar_width * 1), color = "black", size = 2) +
  theme(axis.text.y = element_text(angle = 0, hjust = 1),
        # Set the legend position
        legend.position = c(1, 0),
        # Set the legend anchor point
        legend.justification = c(1, 0),
        text = element_text(size = 10),
        plot.title = element_text(hjust = 0.5)) +
  labs(
    y = "Model",
    x = "F1 Score") +
  scale_fill_discrete(
    name = paste('ID: ', run_id),
    labels = c("Test F1", "Val F1")) +
  coord_cartesian(xlim = c(x_left_lim, x_right_lim))

# Save the plot as a file
print(bar_chart)
new_csv_filename <- paste0(substr(filename, 1, nchar(filename) - 4), "_R.png")
ggsave(new_csv_filename, height = plot_height)
