# ============================================================
# 02_glmm_analysis.R  –  Improved Penalty Shootout GLMM Analysis
# ============================================================
library(recipes)
library(workflows)
library(parsnip)
library(rsample)
library(yardstick)
library(tune)
library(dials)
library(vip)
library(ggplot2)
library(readr)      
library(themis)                     
library(tidymodels)   
library(lme4)         
library(ggplot2)
library(tidyverse)
library(broom.mixed)   
library(broom)         
library(yardstick)
library(dplyr)
library(car)  # For VIF check
library(performance)  # For model diagnostics
library(pROC)
library(caret)  # For confusionMatrix function
dir.create("figures", showWarnings = FALSE)
dir.create("models",  showWarnings = FALSE)

# --- 1. Load Prepared Data -------------------------------------
df <- read_csv("data/shootout_model_data_2.csv", show_col_types = FALSE)

# Exploratory Data Analysis
cat("Exploratory Data Analysis:\n")
cat("Total Sample Size:", nrow(df), "\n")
cat("Target Variable Distribution:\n")
print(table(df$target_win_home))
cat("Target Variable Proportions:\n")
print(prop.table(table(df$target_win_home)))

# Check Distribution of Variables
cat("\nSample Count by Variables:\n")
cat("Confederation Distribution:\n")
print(table(df$home_confed))
cat("Major Tournament Distribution:\n")
print(table(df$big_tourney))
cat("Home/Neutral Distribution:\n")
print(table(df$neutral))

# --- 1.1 Improved Data Preprocessing -------------------------------------
df <- df %>%
  mutate(
    # Target Variable Encoding
    target_win_home = factor(target_win_home, levels = c("FALSE","TRUE")),
    
    # Categorical Variable Processing
    confed = factor(home_confed),
    big_tourney = factor(big_tourney),
    neutral = factor(neutral),
    
    # Continuous Variable Standardization (Using Robust Methods)
    abs_goal_diff_scaled = scale(abs_goal_diff)[,1],
    total_goals_scaled = scale(total_goals)[,1],
    late_goal_shift_scaled = scale(late_goal_shift)[,1],
    pressure_index_scaled = scale(pressure_index)[,1],
    
    # New: Interaction Variables
    neutral_big_tourney = as.numeric(neutral) * as.numeric(big_tourney),
    home_advantage = as.numeric(!neutral),
    
    # New: Non-linear Transformations
    abs_goal_diff_sqrt = sqrt(abs_goal_diff + 1),
    total_goals_log = log(total_goals + 1),
    
    # New: Ratio Variables
    late_goal_ratio = ifelse(total_goals > 0, late_goals / total_goals, 0),
    goal_efficiency = ifelse(abs_goal_diff > 0, total_goals / abs_goal_diff, total_goals)
  )

# Data Quality Check
cat("\nData Quality Check:\n")
cat("Missing Value Count:\n")
print(colSums(is.na(df)))

# Handle Missing Values
df <- df %>%
  mutate(
    across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)),
    across(where(is.character), ~ifelse(is.na(.), names(which.max(table(.))), .))
  )

# --- 2. Improved Train/Test Split -------------------------------------
set.seed(42)
spl <- initial_split(df, strata = target_win_home, prop = 0.8)
train <- training(spl) %>% droplevels()
test  <- testing(spl)

cat("\nTraining Set Sample Size:", nrow(train), "\n")
cat("Test Set Sample Size:", nrow(test), "\n")

# --- 2.1 Align Factor Levels for Test Set --------------------------------
align_levels <- function(new_df, ref_df, vars){
  for (v in vars){
    new_df[[v]] <- factor(new_df[[v]], levels = levels(ref_df[[v]]))
    new_df[[v]][is.na(new_df[[v]])] <- levels(ref_df[[v]])[1]
  }
  new_df
}
test <- align_levels(test, train, c("confed","neutral","big_tourney"))

# --- 3. Model Diagnostics and Variable Selection ----------------------------------------
# First fit a simple model for diagnostics
simple_form <- target_win_home ~ confed + neutral + big_tourney + 
  abs_goal_diff_scaled + late_goal_shift_scaled + (1 | home_team)

simple_model <- glmer(
  simple_form, data = train, family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
)

# Model Diagnostics
cat("\nSimple Model Diagnostics:\n")
print(summary(simple_model))

# Check Convergence
if (!is.null(simple_model@optinfo$conv$lme4$messages)) {
  cat("Warning: Model convergence issues\n")
  print(simple_model@optinfo$conv$lme4$messages)
}

# --- 4. Improved GLMM Formulas (Stepwise Variable Addition) ----------------------------------------
# Stepwise model building to avoid multicollinearity
form1 <- target_win_home ~ confed + neutral + big_tourney + 
  abs_goal_diff_scaled + (1 | home_team)

form2 <- target_win_home ~ confed + neutral + big_tourney + 
  abs_goal_diff_scaled + late_goal_shift_scaled + (1 | home_team)

form3 <- target_win_home ~ confed + neutral + big_tourney + 
  abs_goal_diff_scaled + late_goal_shift_scaled + pressure_index_scaled + (1 | home_team)

form4 <- target_win_home ~ confed + neutral + big_tourney + 
  abs_goal_diff_scaled + late_goal_shift_scaled + pressure_index_scaled + 
  neutral_big_tourney + (1 | home_team)

# Fit multiple models for comparison
models <- list()
models[["basic"]] <- glmer(form1, data = train, family = binomial,
                           control = glmerControl(optimizer = "bobyqa"))

models[["with_late_goals"]] <- glmer(form2, data = train, family = binomial,
                                     control = glmerControl(optimizer = "bobyqa"))

models[["with_pressure"]] <- glmer(form3, data = train, family = binomial,
                                   control = glmerControl(optimizer = "bobyqa"))

models[["full"]] <- glmer(form4, data = train, family = binomial,
                          control = glmerControl(optimizer = "bobyqa"))

# Model Comparison
aic_values <- sapply(models, AIC)
bic_values <- sapply(models, BIC)

cat("\nModel Comparison (AIC):\n")
print(aic_values)
cat("\nModel Comparison (BIC):\n")
print(bic_values)

# Select Best Model
best_model_name <- names(which.min(aic_values))
model_glmm <- models[[best_model_name]]

cat("\nSelected Best Model:", best_model_name, "\n")

# --- 5. Model Diagnostics and Validation ---------------------------------------
cat("\nFinal Model Summary:\n")
print(summary(model_glmm))

# Check Random Effects
ranef_summary <- ranef(model_glmm)
cat("\nRandom Effects Variance:\n")
print(VarCorr(model_glmm))

# Check Fixed Effects
fe_summary <- broom.mixed::tidy(model_glmm, effects = "fixed")
cat("\nFixed Effects:\n")
print(fe_summary)

# --- 6. Prediction and AUC Calculation ---------------------------------------
test <- test %>%
  mutate(pred_prob = predict(model_glmm, newdata = ., type = "response",
                             allow.new.levels = TRUE))

auc <- roc_auc(
  test, truth = target_win_home, pred_prob,
  event_level = "second"
)
cat("\nTest Set AUC =", round(auc$.estimate, 4), "\n")

# --- 7. Fixed Effects OR & CI (Improved Version) -------------------------------
fe_tbl <- broom.mixed::tidy(
  model_glmm, effects = "fixed",
  conf.int = TRUE, exponentiate = TRUE
) %>%
  select(term, OR = estimate, conf.low, conf.high, p.value, std.error) %>%
  mutate(
    significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**", 
      p.value < 0.05 ~ "*",
      TRUE ~ ""
    )
  )

cat("\nFixed Effects Results:\n")
print(fe_tbl)

write_csv(fe_tbl, "figures/glmm_fixed_effects_improved.csv")

# --- 8. Improved Visualizations ----------------------------------------------
# 8.1 Confederation Win Rate (Improved Version)
p_confed <- df %>%
  group_by(confed) %>%
  summarise(
    win_rate = mean(target_win_home == "TRUE"),
    n_matches = n(),
    .groups = "drop"
  ) %>%
  filter(n_matches >= 5) %>%  # Only show confederations with sufficient samples
  ggplot(aes(reorder(confed, win_rate), win_rate)) +
  geom_col(fill = "#2c7fb8") +
  geom_text(aes(label = sprintf("%.1f%%", win_rate*100)), 
            vjust = -0.5, size = 3) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 0.8)) +
  labs(title = "Penalty Shootout Win Rate by Confederation",
       subtitle = "Only confederations with 5+ matches",
       x = NULL, y = "Win Rate") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("figures/confed_winrate_improved.png", plot = p_confed,
       width = 8, height = 6)

# 8.2 Home Advantage and Event Size (Improved Version)
p_home <- df %>%
  mutate(
    neutral = fct_recode(neutral, "Home" = "FALSE", "Neutral" = "TRUE"),
    big_tourney = fct_recode(big_tourney, "Small Event" = "FALSE", "Major Event" = "TRUE")
  ) %>%
  group_by(neutral, big_tourney) %>%
  summarise(
    win_rate = mean(target_win_home == "TRUE"),
    n_matches = n(),
    .groups = "drop"
  ) %>%
  ggplot(aes(neutral, win_rate, fill = big_tourney)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", win_rate*100)), 
            position = position_dodge(width = 0.7), vjust = -0.5, size = 3) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 0.8)) +
  labs(title = "Home Advantage and Event Size",
       x = NULL, y = "Win Rate", fill = "Event Type") +
  theme_minimal()

ggsave("figures/home_big_effect_improved.png", plot = p_home,
       width = 8, height = 6)

# 8.3 New: Goal Difference vs Win Rate Relationship
p_goal_diff <- df %>%
  group_by(abs_goal_diff) %>%
  summarise(
    win_rate = mean(target_win_home == "TRUE"),
    n_matches = n(),
    .groups = "drop"
  ) %>%
  filter(n_matches >= 3) %>%  # Only show goal differences with sufficient samples
  ggplot(aes(factor(abs_goal_diff), win_rate)) +
  geom_col(fill = "#e41a1c") +
  geom_text(aes(label = sprintf("%.1f%%", win_rate*100)), 
            vjust = -0.5, size = 3) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 0.8)) +
  labs(title = "Goal Difference and Penalty Shootout Win Rate",
       x = "90-minute Goal Difference", y = "Win Rate") +
  theme_minimal()

ggsave("figures/goal_diff_winrate.png", plot = p_goal_diff,
       width = 8, height = 6)

# --- 9. Model Performance Evaluation -----------------------------------------
# -------------------- 6. Model Prediction --------------------
glmm_probs <- test$pred_prob
glmm_class <- ifelse(glmm_probs >= 0.5, "TRUE", "FALSE") %>% factor(levels = c("FALSE", "TRUE"))

# -------------------- 7. Calculate AUC --------------------
# Ensure target_win_home is factor with levels c("FALSE", "TRUE")
test$target_win_home <- factor(test$target_win_home, levels = c("FALSE", "TRUE"))

# Build 0/1 labels: TRUE → 1, FALSE → 0
truth <- ifelse(test$target_win_home == "TRUE", 1, 0)

# Use pROC to calculate AUC
auc_score <- roc(response = truth, predictor = glmm_probs, quiet = TRUE)
print(paste("AUC =", round(auc(auc_score), 3)))

# Set output file path (modifiable)
png("figures/glmm_roc_curve.png", width = 800, height = 600)

# Plot and add AUC value to title
plot(auc_score,
     col = "blue",
     main = paste("ROC Curve - GLMM (AUC =", round(auc(auc_score), 3), ")"))

# Close graphics device
dev.off()

# Save AUC value for later use
auc_val <- auc(auc_score)

# 同时保留ggplot2版本用于一致性
roc_data <- roc_curve(test, target_win_home, pred_prob)
p_roc <- ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "#e41a1c", linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "gray50") +
  labs(title = "GLMM ROC Curve (ggplot2)",
       subtitle = sprintf("AUC = %.4f", auc$.estimate),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("figures/glmm_roc_curve_ggplot.png", p_roc, width = 7, height = 6, dpi = 300)

# Threshold Optimization
thresholds <- seq(0.1, 0.9, by = 0.05)
threshold_metrics <- map_dfr(thresholds, function(thr) {
  pred_class <- ifelse(test$pred_prob >= thr, "TRUE", "FALSE")
  actual_class <- as.character(test$target_win_home)
  
  conf_mat <- table(
    Actual = factor(actual_class, levels = c("FALSE", "TRUE")),
    Predicted = factor(pred_class, levels = c("FALSE", "TRUE"))
  )
  
  tp <- conf_mat["TRUE", "TRUE"]
  tn <- conf_mat["FALSE", "FALSE"]
  fp <- conf_mat["FALSE", "TRUE"]
  fn <- conf_mat["TRUE", "FALSE"]
  
  accuracy <- (tp + tn) / sum(conf_mat)
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
  f1_score <- ifelse(precision + recall > 0, 2 * (precision * recall) / (precision + recall), 0)
  
  tibble(
    threshold = thr,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score
  )
})

# Find optimal threshold (based on F1 score)
optimal_threshold <- threshold_metrics$threshold[which.max(threshold_metrics$f1_score)]
cat("Optimal threshold based on F1 score:", optimal_threshold, "\n")

# Use optimal threshold for classification prediction
pred_class <- ifelse(test$pred_prob >= optimal_threshold, "TRUE", "FALSE")
actual_class <- as.character(test$target_win_home)

# Confusion matrix object
conf_mat <- confusionMatrix(factor(pred_class, levels = c("FALSE", "TRUE")), 
                           factor(actual_class, levels = c("FALSE", "TRUE")))
print(conf_mat)

# Convert to data.frame for plotting
conf_tbl <- as.data.frame(conf_mat$table)
names(conf_tbl) <- c("Prediction", "Reference", "Freq")

# Draw heatmap
p_conf <- ggplot(conf_tbl, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(title = "Confusion Matrix - GLMM",
       x = "Predicted Label",
       y = "True Label") +
  theme_minimal(base_size = 14)

ggsave("figures/glmm_confusion_matrix.png", p_conf, width = 6, height = 5)

# Threshold optimization results visualization
p_threshold <- ggplot(threshold_metrics, aes(x = threshold)) +
  geom_line(aes(y = accuracy, color = "Accuracy"), linewidth = 1) +
  geom_line(aes(y = precision, color = "Precision"), linewidth = 1) +
  geom_line(aes(y = recall, color = "Recall"), linewidth = 1) +
  geom_line(aes(y = f1_score, color = "F1 Score"), linewidth = 1.5) +
  geom_vline(xintercept = optimal_threshold, linetype = "dashed", color = "red") +
  scale_color_manual(values = c("Accuracy" = "#377eb8", "Precision" = "#4daf4a", 
                                "Recall" = "#ff7f00", "F1 Score" = "#e41a1c")) +
  labs(title = "GLMM Threshold Optimization",
       subtitle = sprintf("Optimal threshold: %.2f", optimal_threshold),
       x = "Threshold", y = "Metric Value", color = "Metrics") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("figures/glmm_threshold_optimization.png", p_threshold, width = 8, height = 6, dpi = 300)

# --- 10. Save Model and Report -----------------------------------------
saveRDS(model_glmm, "models/pk_glmm_improved.rds")

# Create Performance Report
perf_report <- tibble(
  Model = "Penalty Shootout Prediction GLMM Model (Improved Version)",
  AUC_pROC = round(auc_val, 3),
  AUC_yardstick = round(auc$.estimate, 4),
  Optimal_Threshold = round(optimal_threshold, 3),
  Accuracy = round(conf_mat$overall["Accuracy"], 4),
  Precision = round(conf_mat$byClass["Precision"], 4),
  Recall = round(conf_mat$byClass["Recall"], 4),
  F1_Score = round(conf_mat$byClass["F1"], 4),
  Sample_Size = nrow(df),
  Training_Sample_Size = nrow(train),
  Test_Sample_Size = nrow(test),
  Significant_Variables = sum(fe_tbl$p.value < 0.05),
  Model_Complexity = length(fe_tbl$term)
)
write_csv(perf_report, "figures/glmm_performance_report.csv")

cat("\n GLMM Analysis Complete!\n")
cat("Test Set AUC (pROC) =", round(auc_val, 3), "\n")
cat("Test Set AUC (yardstick) =", round(auc$.estimate, 4), "\n")
cat("Significant Variables =", sum(fe_tbl$p.value < 0.05), "\n")