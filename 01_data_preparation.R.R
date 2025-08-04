# ============================================================
# 01_data_preparation.R  –  Improved Penalty Shootout Data Preprocessing
# ============================================================
library(tidyverse)
library(lubridate)
library(janitor)

# --- 1. Load Raw Data ----------------------------------------
results   <- read_csv("data/results.csv")      %>% clean_names()
goals     <- read_csv("data/goalscorers.csv")  %>% clean_names()
shootouts <- read_csv("data/shootouts.csv")    %>% clean_names()

# --- 2. Improved Derived Core Features ----------------------------------------
results <- results %>%
  mutate(
    date        = as.Date(date),
    match_key   = paste(date, home_team, away_team),
    goal_diff   = home_score - away_score,
    total_goals = home_score + away_score,
    abs_goal_diff = abs(goal_diff),
    ninety_winner = case_when(
      home_score > away_score ~ home_team,
      home_score < away_score ~ away_team,
      TRUE ~ "Draw"
    ),
    # New: Match Type Classification
    match_type = case_when(
      home_score == away_score ~ "draw",
      abs_goal_diff == 1 ~ "close",
      abs_goal_diff == 2 ~ "moderate", 
      TRUE ~ "clear"
    ),
    # New: Goal Efficiency
    goal_efficiency = ifelse(total_goals > 0, total_goals / (abs_goal_diff + 1), 0)
  )

# Improved Goal Time Analysis
goals_summary <- goals %>%
  mutate(
    date      = as.Date(date),
    match_key = paste(date, home_team, away_team),
    # More Refined Time Period Division
    period    = case_when(
      minute <= 30 ~ "early",
      minute <= 60 ~ "mid", 
      minute <= 90 ~ "late",
      TRUE ~ "extra"
    )
  ) %>%
  group_by(match_key, team) %>%
  summarise(
    early_goals = sum(period == "early"),
    mid_goals = sum(period == "mid"),
    late_goals = sum(period == "late"),
    extra_goals = sum(period == "extra"),
    total_team_goals = n(),
    .groups = "drop"
  ) %>%
  mutate(
    # New: Goal Time Distribution Features
    late_goal_ratio = ifelse(total_team_goals > 0, late_goals / total_team_goals, 0),
    early_goal_ratio = ifelse(total_team_goals > 0, early_goals / total_team_goals, 0),
    goal_momentum = late_goals - early_goals  # Late Goal Advantage
  )

results_mini <- results %>%
  select(match_key, home_team, away_team, goal_diff, total_goals, abs_goal_diff,
         tournament, neutral, country, ninety_winner, match_type, goal_efficiency)

# --- 3. Improved Merge to shootout_df -----------------------------------
shootout_df <- shootouts %>%
  mutate(
    date      = as.Date(date),
    match_key = paste(date, home_team, away_team)
  ) %>%
  left_join(results_mini, by = "match_key", suffix = c(".shoot", ".res")) %>%
  left_join(goals_summary, by = c("match_key", "winner" = "team")) %>%
  rename(home_team = home_team.shoot,
         away_team = away_team.shoot) %>%
  mutate(
    # Basic Features
    draw_90          = goal_diff == 0,
    winner_is_ninety_winner = winner == ninety_winner,
    
    # Improved Goal Time Features
    late_goal_shift  = late_goals - early_goals,
    late_goal_ratio  = ifelse(!is.na(late_goal_ratio), late_goal_ratio, 0),
    early_goal_ratio = ifelse(!is.na(early_goal_ratio), early_goal_ratio, 0),
    goal_momentum    = ifelse(!is.na(goal_momentum), goal_momentum, 0),
    
    # Match Intensity Features
    goals_penalty_ratio = total_goals / (abs_goal_diff + 1),
    
    # Improved Major Tournament Classification (Create this variable first)
    big_tourney = tournament %in% c(
      "FIFA World Cup", "UEFA Euro", "Copa América",
      "AFC Asian Cup", "African Cup of Nations", "Olympics Men"
    ),
    
    # Now can safely use big_tourney variable
    match_intensity = total_goals * as.numeric(big_tourney),
    
    # Improved Confederation Classification (More Comprehensive Mapping)
    home_confed = case_when(
      home_team %in% c("Brazil","Argentina","Uruguay","Chile","Colombia","Peru","Paraguay","Ecuador","Venezuela","Bolivia") ~ "CONMEBOL",
      home_team %in% c("Germany","France","Italy","Spain","England","Netherlands","Belgium","Portugal","Switzerland","Sweden","Denmark","Norway","Poland","Czech Republic","Hungary","Austria","Croatia","Serbia","Slovenia","Slovakia","Ukraine","Russia","Turkey","Greece","Romania","Bulgaria","Finland","Iceland","Estonia","Latvia","Lithuania","Luxembourg","Malta","Cyprus","Albania","Montenegro","Bosnia and Herzegovina","North Macedonia","Kosovo","Moldova","Georgia","Armenia","Azerbaijan","Kazakhstan","Uzbekistan","Kyrgyzstan","Tajikistan","Turkmenistan") ~ "UEFA",
      home_team %in% c("USA","Mexico","Canada","Costa Rica","Honduras","El Salvador","Guatemala","Panama","Nicaragua","Belize","Jamaica","Trinidad and Tobago","Haiti","Cuba","Dominican Republic","Puerto Rico","Grenada","St. Vincent and the Grenadines","Barbados","Antigua and Barbuda","St. Kitts and Nevis","Dominica","St. Lucia") ~ "CONCACAF",
      home_team %in% c("Japan","South Korea","China","Iran","Saudi Arabia","Australia","Qatar","UAE","Iraq","Syria","Lebanon","Jordan","Oman","Kuwait","Bahrain","Yemen","Palestine","Mongolia","North Korea","Vietnam","Thailand","Myanmar","Laos","Cambodia","Malaysia","Singapore","Indonesia","Philippines","Brunei","Timor-Leste","Nepal","Bhutan","Bangladesh","Sri Lanka","Maldives","Pakistan","Afghanistan","Kyrgyzstan","Tajikistan","Turkmenistan","Uzbekistan","Kazakhstan","Azerbaijan","Georgia","Armenia") ~ "AFC",
      home_team %in% c("Nigeria","Ghana","Senegal","Cameroon","Ivory Coast","Morocco","Tunisia","Algeria","Egypt","South Africa","Kenya","Uganda","Tanzania","Ethiopia","Sudan","South Sudan","Eritrea","Djibouti","Somalia","Chad","Niger","Mali","Burkina Faso","Guinea","Guinea-Bissau","Sierra Leone","Liberia","Togo","Benin","Central African Republic","Congo","DR Congo","Gabon","Equatorial Guinea","São Tomé and Príncipe","Angola","Zambia","Zimbabwe","Botswana","Namibia","Lesotho","Eswatini","Madagascar","Comoros","Mauritius","Seychelles","Burundi","Rwanda","Malawi","Mozambique") ~ "CAF",
      TRUE ~ "Other"
    ),
    
    # New: Match Pressure Features
    pressure_index = as.numeric(big_tourney) * (1 + abs_goal_diff),
    home_advantage = as.numeric(!neutral),
    neutral_big_tourney = as.numeric(neutral) * as.numeric(big_tourney),
    
    # New: Match Balance Features
    game_balance = 1 / (1 + abs_goal_diff),
    
    # New: Time Features
    year = year(date),
    decade = floor(year / 10) * 10,
    
    # New: Match Result Type
    result_type = case_when(
      draw_90 ~ "draw",
      abs_goal_diff == 1 ~ "close_win",
      abs_goal_diff == 2 ~ "moderate_win",
      TRUE ~ "clear_win"
    ),
    
    # Type Conversion
    neutral         = as.logical(neutral),
    draw_90         = as.logical(draw_90),
    big_tourney     = as.logical(big_tourney),
    target_win_home = factor(winner == home_team, levels = c(FALSE, TRUE))
  )

# Data Quality Check
cat("Data Quality Report:\n")
cat("Total Sample Size:", nrow(shootout_df), "\n")
cat("Missing Value Check:\n")
print(colSums(is.na(shootout_df)))
cat("Target Variable Distribution:\n")
print(table(shootout_df$target_win_home))
cat("Major Tournament Sample Count:\n")
print(table(shootout_df$big_tourney))

# Handle Missing Values
shootout_df <- shootout_df %>%
  mutate(
    # Fill numeric variables with median
    across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)),
    # Fill categorical variables with mode
    across(where(is.character), ~ifelse(is.na(.), names(which.max(table(.))), .))
  )

write_csv(shootout_df, "data/shootout_model_data_2.csv")
message("proved shootout_model_data_2.csv saved | Sample size = ", nrow(shootout_df))
message("new features: Match intensity, Pressure index, Time features, Improved confederation classification, etc.")

