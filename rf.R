#install.packages(c("randomForest", "pROC", "caret", "dplyr"))
library(randomForest)
library(pROC)
library(caret)
library(dplyr)
library(ggplot2)
library(ranger)

df <- read.csv("data/shootout_model_data_2.csv")  # Replace with your file path
# -------------------- 3. Data Preprocessing --------------------
# Remove columns that don't contribute to model training (such as dates and unique identifiers)
df_model <- df %>%
  select(-c(date, match_key, home_team, away_team, winner, first_shooter,
            home_team.res, away_team.res, country)) %>%
  na.omit()

# Convert character columns to factors
df_model <- df_model %>%
  mutate(across(where(is.character), as.factor))

# Ensure target variable is binary factor (TRUE/FALSE)
df_model$target_win_home <- as.factor(df_model$target_win_home)

# -------------------- 4. Split Training and Test Sets --------------------
set.seed(42)
split_idx <- createDataPartition(df_model$target_win_home, p = 0.8, list = FALSE)
train_data <- df_model[split_idx, ]
test_data  <- df_model[-split_idx, ]

# -------------------- 5. Train Random Forest Model Using Ranger --------------------
rf_model <- ranger(
  formula = target_win_home ~ .,
  data = train_data,
  num.trees = 100,
  probability = TRUE,   # Return probability predictions to calculate AUC
  seed = 42
)

# -------------------- 6. Model Prediction --------------------
rf_pred <- predict(rf_model, data = test_data)
rf_probs <- rf_pred$predictions[, 2]  # Second column is P(class = TRUE)
rf_class <- ifelse(rf_probs >= 0.5, "TRUE", "FALSE") %>% factor(levels = c("FALSE", "TRUE"))

# -------------------- 7. Calculate AUC --------------------
# Ensure target_win_home is factor with levels c("FALSE", "TRUE")
test_data$target_win_home <- factor(test_data$target_win_home, levels = c("FALSE", "TRUE"))

# Build 0/1 labels: TRUE → 1, FALSE → 0
truth <- ifelse(test_data$target_win_home == "TRUE", 1, 0)

# Use pROC to calculate AUC
auc_score <- roc(response = truth, predictor = rf_probs, quiet = TRUE)
print(paste("AUC =", round(auc(auc_score), 3)))


# Set output file path (modifiable)
png("figures/roc_ranger_rf.png", width = 800, height = 600)

# Plot and add AUC value to title
plot(auc_score,
     col = "blue",
     main = paste("ROC Curve - ranger RF (AUC =", round(auc(auc_score), 3), ")"))

# Close graphics device
dev.off()

# Predicted classes (converted using 0.5 threshold)
rf_class <- ifelse(rf_probs >= 0.5, "TRUE", "FALSE") %>%
  factor(levels = c("FALSE", "TRUE"))  # Ensure consistent order

# True labels
true_class <- factor(test_data$target_win_home, levels = c("FALSE", "TRUE"))

# Confusion matrix object
conf_mat <- confusionMatrix(rf_class, true_class)
print(conf_mat)
# Convert to data.frame for plotting
conf_tbl <- as.data.frame(conf_mat$table)
names(conf_tbl) <- c("Prediction", "Reference", "Freq")

# Draw heatmap
ggplot(conf_tbl, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(title = "Confusion Matrix - ranger RF",
       x = "Predicted Label",
       y = "True Label") +
  theme_minimal(base_size = 14)
ggsave("figures/confusion_matrix_ranger_rf.png", width = 6, height = 5)

