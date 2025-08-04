library(xgboost)
library(caret)
library(dplyr)
library(Matrix)
library(ggplot2)

# ---------- Load Data ----------
df <- read.csv("data/shootout_model_data_2.csv")  # Modify to actual path

# ---------- Keep Features Except Target Column ----------
target_col <- ncol(df)
target <- df[[target_col]]
features <- df[, -target_col]

# ---------- Convert Character Columns to Factors ----------
features <- features %>%
  mutate(across(where(is.character), as.factor))

# ---------- Merge and Handle Missing Values ----------
data <- cbind(features, target = target)
data <- na.omit(data)

# ---------- Extract Features and Target Again ----------
y <- data$target
X <- data[, setdiff(colnames(data), "target")]
# Use sparse.model.matrix for one-hot encoding
X_sparse <- sparse.model.matrix(~ . - 1, data = X)

# Binary classification labels must be 0 / 1
y <- as.numeric(as.factor(y)) - 1  # Convert to 0/1
set.seed(42)
train_idx <- sample(1:nrow(X_sparse), 0.8 * nrow(X_sparse))
dtrain <- xgb.DMatrix(data = X_sparse[train_idx, ], label = y[train_idx])
dtest  <- xgb.DMatrix(data = X_sparse[-train_idx, ], label = y[-train_idx])


params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.3
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, eval = dtest),
  verbose = 0
)
# Predict Probabilities
pred_prob <- predict(xgb_model, dtest)
pred_label <- ifelse(pred_prob >= 0.5, 1, 0)

library(pROC)
roc_obj <- roc(y[-train_idx], pred_prob)
auc_val <- auc(roc_obj)

png("figures/xgboost_roc.png", width = 800, height = 600)
plot(roc_obj, col = "darkorange", main = paste("XGBoost ROC Curve (AUC =", round(auc_val, 3), ")"))
dev.off()

# Confusion Matrix Object
# Predicted labels (1 = positive class, 0 = negative class)
pred_label <- ifelse(pred_prob >= 0.5, 1, 0)

# Actual labels
actual_label <- y[-train_idx]

# Create factors, ensure consistent order
pred_factor <- factor(pred_label, levels = c(0, 1))
actual_factor <- factor(actual_label, levels = c(0, 1))

# Get confusion matrix
conf_mat <- confusionMatrix(pred_factor, actual_factor)
print(conf_mat)
# Convert to data.frame for plotting
conf_tbl <- as.data.frame(conf_mat$table)
names(conf_tbl) <- c("Prediction", "Reference", "Freq")
# Draw heatmap and assign to p_conf
p_conf <- ggplot(conf_tbl, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(title = "Confusion Matrix - XGBoost",
       x = "Predicted Label",
       y = "True Label") +
  theme_minimal(base_size = 14)

# Display image (optional)
print(p_conf)

# Save image
ggsave("figures/confusion_matrix_ranger_XGBoost.png", p_conf, width = 6, height = 5)


