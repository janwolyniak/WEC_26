

setwd("C:/Users/macku/OneDrive/Dokumenty/WWEECC/WEC_26/data")
df <- read.csv("players_quarters_final.csv")
library(tidyverse)
library(skimr) 
library(tidyverse)
library(sandwich)
library(lmtest)

###### LOGIT MVP #######

# Ensure the dependent variable is strictly 0 and 1
df <- df %>%
  mutate(scored_after_binary = as.numeric(scored_after))

# 2. Estimate the Standard Pooled Logit Model
logit_mvp <- glm(
  formula = scored_after_binary ~ last15_hsr + last15_sprints  + last15_peak_speed + last15_shots + 
  I(cumul_shots - last15_shots) + cumul_distance + cumul_mean_max_speed 
  + as.factor(position),
  data = df,
  family = binomial(link = "logit")
)

logit_mvp <- glm(
  formula = scored_after_binary ~ last15_hsr + cumul_distance,
  data = df,
  family = binomial(link = "logit")
)

# 3. Calculate Clustered Standard Errors
clustered_vcov <- vcovCL(logit_mvp, cluster = ~ player_appearance_id)

# 4. Generate the Final Results Table
robust_results <- coeftest(logit_mvp, vcov = clustered_vcov)
print(robust_results)

# 5. Calculate Odds Ratios for easier interpretation
cat("\n=== ODDS RATIOS ===\n")
# Exponentiating the coefficients gives us the Odds Ratios
odds_ratios <- exp(coef(robust_results))
print(odds_ratios)



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
