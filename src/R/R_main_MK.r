

setwd("C:/Users/macku/OneDrive/Dokumenty/WWEECC/WEC_26/data")
library(tidyverse)
library(skimr) 
library(tidyverse)
library(sandwich)
library(lmtest)
library(car)
library(stargazer)

xd<- read_csv("players_quarters_final_step1.csv", show_col_types = FALSE)

df <- read_csv("knn_dataset.csv", show_col_types = FALSE) %>%
  mutate(
    fixture_id = as.numeric(str_extract(scored_after_eval_key, "^\\d+")),
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
    
    # Calculate history for NEW passing variables
    history_pass_passed = cumul_pass_passed - last15_pass_passed,
    history_pass_passed_accurate = cumul_pass_passed_accurate - last15_pass_passed_accurate,
    history_pass_received = cumul_pass_received - last15_pass_received,
    history_pass_received_accurate = cumul_pass_received_accurate - last15_pass_received_accurate
  ) %>%
  # Exclude goalkeepers as you mentioned
  filter(position != "G") %>%
  drop_na()

##Testowy: - 1169, 1190, 1198, 1215, 1216, 1237 
# Split the dataframe using fixture_id
test_match_ids <- c(1169, 1190, 1198, 1215, 1216, 1237)
df_train <- df %>% filter(!(fixture_id %in% test_match_ids))
df_test  <- df %>% filter(fixture_id %in% test_match_ids)

# Print a summary to verify the split was successful
cat("\n--- Dataset Split Summary ---\n")
cat("Training Set Rows:", nrow(df_train), "\n")
cat("Testing Set Rows :", nrow(df_test), "\n\n")

###### LOGIT MVP #######

# 2. Estimate the Standard Pooled Logit Model
logit_bloated <- glm(
  formula = scored_after_binary ~ 
    # A. Match Context
    position + is_home + checkpoint + minutes_in_game +
    
    # B. Recent Form (Last 15)
    last15_shots + last15_shots_on_target + last15_shots_under_press + last15_shots_top_third +
    last15_hsr + last15_sprints  +
    last15_pass_received + last15_pass_passed + last15_pass_received_accurate + last15_pass_passed_accurate +
    
    # C. Historical Form (Prior to Last 15)
    history_shots + history_shots_on_target + history_shots_under_press + history_shots_top_third +
    history_hsr + history_sprints +
    history_pass_passed + history_pass_passed_accurate + history_pass_received + history_pass_received_accurate +
    
    # D. Sensible Interactions
    position:last15_shots +                  # Does a shot from a Forward mean more than from a Defender?
    last15_hsr:last15_shots +                # High physical intensity combined with attacking product
    is_home:last15_shots +                   # Does home advantage improve shot danger?
    position:last15_pass_received_accurate + # Forwards receiving accurate passes vs Defenders
    last15_hsr:last15_pass_passed_accurate,  # High running intensity while maintaining accurate passing
    
  data = df_train,
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


##Balanced ACCURACY##
# Predict probabilities on the unseen test dataset
df_test$predicted_probability <- predict(logit_bloated, newdata = df_test, type = "response")

# Convert probabilities to a binary prediction. 
# 0.5 is standard, but you can lower it for rare events like scoring.
threshold <- 0.05 
df_test$predicted_class <- ifelse(df_test$predicted_probability >= threshold, 1, 0)

# Generate a Confusion Matrix ensuring both 0 and 1 levels are present
conf_matrix <- table(
  Predicted = factor(df_test$predicted_class, levels = c(0, 1)), 
  Actual = factor(df_test$scored_after_binary, levels = c(0, 1))
)

print(conf_matrix)

# Extract metrics from the 2x2 confusion matrix
TN <- conf_matrix[1, 1] # True Negatives
FN <- conf_matrix[1, 2] # False Negatives
FP <- conf_matrix[2, 1] # False Positives
TP <- conf_matrix[2, 2] # True Positives

# Calculate Sensitivity (TPR) and Specificity (TNR)
sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)

# Calculate Balanced Accuracy
(balanced_accuracy <- (sensitivity + specificity) / 2)



##PUBLICATION TABLES
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
      P_Value < 0.1 ~ ".",
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



