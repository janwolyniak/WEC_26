

setwd("C:/Users/macku/OneDrive/Dokumenty/WWEECC/WEC_26/data")
library(tidyverse)
library(skimr) 
library(tidyverse)
library(sandwich)
library(lmtest)
library(car)
library(stargazer)

df <- read_csv("players_quarters_final_step1.csv", show_col_types = FALSE) %>%
  mutate(
    scored_after_binary = as.numeric(scored_after),
    position = as.factor(position),
    is_home = as.factor(is_home),
    checkpoint = as.factor(checkpoint), # Crucial for 'time remaining' effects
    
    # Calculate history for original overlapping variables
    history_shots = cumul_shots - last15_shots,
    history_shots_on_target = cumul_shots_on_target - last15_shots_on_target,
    history_shots_under_press = cumul_shots_under_press - last15_shots_under_press,
    history_shots_top_third = cumul_shots_top_third - last15_shots_top_third,
    history_hsr = cumul_hsr - last15_hsr,
    history_sprints = cumul_sprints - last15_sprints,
    history_distance = cumul_distance - last15_distance,
    
    # Calculate history for NEW passing variables
    history_pass_passed = cumul_pass_passed - last15_pass_passed,
    history_pass_passed_accurate = cumul_pass_passed_accurate - last15_pass_passed_accurate,
    history_pass_received = cumul_pass_received - last15_pass_received,
    history_pass_received_accurate = cumul_pass_received_accurate - last15_pass_received_accurate
  ) %>%
  # Exclude goalkeepers as you mentioned
  filter(position != "G") %>%
  # THE FIX: Drop any rows with NA values so the step() function has a constant sample size
  drop_na()


###### LOGIT MVP #######

# 2. Estimate the Standard Pooled Logit Model
logit_bloated <- glm(
  formula = scored_after_binary ~ 
    # A. Match Context
    position + is_home + checkpoint + minutes_in_game +
    
    # B. Recent Form (Last 15)
    last15_shots + last15_shots_on_target + last15_shots_under_press + last15_shots_top_third +
    last15_hsr + last15_sprints + last15_distance + last15_peak_speed +
    last15_pass_received + last15_pass_passed + last15_pass_received_accurate + last15_pass_passed_accurate +
    
    # C. Historical Form (Prior to Last 15)
    history_shots + history_shots_on_target + history_shots_under_press + history_shots_top_third +
    history_hsr + history_sprints + cumul_peak_speed +
    history_pass_passed + history_pass_passed_accurate + history_pass_received + history_pass_received_accurate +
    
    # D. Sensible Interactions
    position:last15_shots +                  # Does a shot from a Forward mean more than from a Defender?
    last15_hsr:last15_shots +                # High physical intensity combined with attacking product
    is_home:last15_shots +                   # Does home advantage improve shot danger?
    position:last15_pass_received_accurate + # Forwards receiving accurate passes vs Defenders
    minutes_in_game:history_distance +       # Compounding fatigue effect
    last15_hsr:last15_pass_passed_accurate,  # High running intensity while maintaining accurate passing
    
  data = df,
  family = binomial(link = "logit")
)
print(logit_bloated)
logit_automated <- step(logit_bloated, direction = "backward", trace = 1)
print(logit_automated)
clustered_vcov_automated <- vcovCL(logit_automated, cluster = ~ player_appearance_id)
robust_results_automated <- coeftest(logit_automated, vcov = clustered_vcov_automated)
print(robust_results_automated)


# 5. Calculate Odds Ratios for easier interpretation
# Exponentiating the coefficients gives us the Odds Ratios
odds_ratios <- exp(coef(robust_results_automated))
print(odds_ratios)


robust_se <- sqrt(diag(clustered_vcov_automated))
stargazer(
  logit_automated, 
  type = "html", 
  se = list(robust_se), # Inject our clustered standard errors
  title = "Goal-Scoring Probability: Pooled Logit",
  dep.var.labels = "Scored After (1=Yes)",
  star.cutoffs = c(0.05, 0.01, 0.001),
  no.space = TRUE,
  digits = 3
)

odds_ratio_table <- data.frame(
  Variable = rownames(robust_results_automated),
  Odds_Ratio = round(exp(robust_results_automated[, "Estimate"]), 3),
  Robust_SE = round(robust_results_automated[, "Std. Error"], 4),
  P_Value = round(robust_results_automated[, "Pr(>|z|)"], 4)
) %>%
  mutate(
    Significance = case_when(
      P_Value < 0.001 ~ "***",
      P_Value < 0.01 ~ "**",
      P_Value < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  arrange(P_Value) # Sorts the table so your best predictors are at the top!

# Print the clean dataframe as a nice HTML table
library(knitr)
library(kableExtra)

odds_ratio_table <- data.frame(
  Variable = rownames(robust_results_automated),
  Odds_Ratio = round(exp(robust_results_automated[, "Estimate"]), 3),
  Robust_SE = round(robust_results_automated[, "Std. Error"], 4),
  P_Value = round(robust_results_automated[, "Pr(>|z|)"], 4)
) %>%
  mutate(
    Significance = case_when(
      P_Value < 0.001 ~ "***",
      P_Value < 0.01 ~ "**",
      P_Value < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  arrange(P_Value) # Sorts the table so your best predictors are at the top!

# Print the clean dataframe as a nice publication-style HTML table
odds_ratio_table %>%
  kbl(format = "html", 
      caption = "Odds Ratios & Significance",
      align = "c") %>% # align = "c" centers all columns and values
  kable_classic(full_width = FALSE, 
                position = "left", 
                html_font = "serif") %>% # kable_classic adds traditional publication horizontal lines
  cat(file = "odds_ratio_table.html")




########





###### DATA VALIDATION CHECKS ######

cat("\n====================================================\n")
cat("3. MISSING VALUES CHECK\n")
cat("====================================================\n")
missing_summary <- df %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "missing_count") %>%
  filter(missing_count > 0) %>%
  mutate(missing_percentage = round((missing_count / nrow(df)) * 100, 2))

if (nrow(missing_summary) == 0) {
  cat("Great news! There are no missing values in the dataset.\n")
} else {
  cat("Missing values found in the following columns:\n")
  print(missing_summary)
}

cat("\n====================================================\n")
cat("4. DUPLICATES CHECK\n")
cat("====================================================\n")
# Each row should uniquely represent one player at one specific checkpoint
duplicates <- df %>%
  group_by(player_appearance_id, checkpoint) %>%
  filter(n() > 1)

cat("Number of duplicate records (based on player_appearance_id + checkpoint):", nrow(duplicates), "\n")

cat("\n====================================================\n")
cat("5. LOGICAL COHERENCE CHECKS\n")
cat("====================================================\n")

# A. Minute coherence (minute_in should be strictly less than minute_out)
invalid_minutes <- df %>% filter(minute_in >= minute_out)
cat("Rows where 'minute_in' >= 'minute_out':", nrow(invalid_minutes), "\n")

# B. Cumulative vs Last 15 minutes metrics coherence
# Cumulative metrics should always be greater than or equal to the "last 15" metrics
logic_sprints <- df %>% filter(cumul_sprints < last15_sprints)
cat("Rows where cumul_sprints < last15_sprints:", nrow(logic_sprints), "\n")

logic_hsr <- df %>% filter(cumul_hsr < last15_hsr)
cat("Rows where cumul_hsr < last15_hsr:", nrow(logic_hsr), "\n")

logic_distance <- df %>% filter(cumul_distance < last15_distance)
cat("Rows where cumul_distance < last15_distance:", nrow(logic_distance), "\n")

logic_shots <- df %>% filter(cumul_shots < last15_shots)
cat("Rows where cumul_shots < last15_shots:", nrow(logic_shots), "\n")

cat("\n====================================================\n")
cat("6. TARGET VARIABLE DISTRIBUTION\n")
cat("====================================================\n")
# Checking class imbalance for the dependent variable 'scored_after'
target_dist <- df %>%
  count(scored_after) %>%
  mutate(
    percentage = round((n / sum(n)) * 100, 2)
  )

print(target_dist)
cat("\nNote: You are likely dealing with a highly imbalanced dataset, which is common in football goal prediction.\n")

cat("\n====================================================\n")
cat("7. CHECKPOINT DISTRIBUTION\n")
cat("====================================================\n")
checkpoint_dist <- df %>%
  count(checkpoint, checkpoint_period, checkpoint_min) %>%
  arrange(checkpoint_period, checkpoint_min)

print(checkpoint_dist)
