# =========================================================
# A.2 Final Analysis: SaaS Churn Prediction
#     Model Comparison & Retention Decision-Support Framework
#
# Project : RavenStack — From Predictive Modeling to
#           Productized Retention Intelligence
# Author  : Genki Hirayama | Northwestern University
# =========================================================

# ── 0. Configuration ──────────────────────────────────────
BASE_DIR <- "C:/Users/Genki/Documents/Homework/AI_Capstone"
DATA_DIR <- file.path(BASE_DIR, "Assignment 1")
OUT_DIR  <- file.path(BASE_DIR, "Assignment 2", "reports")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

set.seed(42)

# ── 1. Install & Load Libraries ───────────────────────────
pkgs <- c("data.table", "dplyr", "lubridate", "ggplot2", "tidyr",
          "broom", "tibble", "randomForest", "xgboost", "pROC", "caret")
invisible(lapply(pkgs, function(p) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}))

# ── 2. Load Data ──────────────────────────────────────────
cat("Loading data...\n")
accounts <- fread(file.path(DATA_DIR, "ravenstack_accounts.csv"))
subs     <- fread(file.path(DATA_DIR, "ravenstack_subscriptions.csv"))
usage    <- fread(file.path(DATA_DIR, "ravenstack_feature_usage.csv"))
tickets  <- fread(file.path(DATA_DIR, "ravenstack_support_tickets.csv"))
churn_ev <- fread(file.path(DATA_DIR, "ravenstack_churn_events.csv"))

cat(sprintf("Loaded: accounts=%d | subs=%d | usage=%d | tickets=%d | churn_events=%d\n",
            nrow(accounts), nrow(subs), nrow(usage), nrow(tickets), nrow(churn_ev)))

# ── 3. Parse Dates ────────────────────────────────────────
accounts[, signup_date := as.Date(signup_date)]
subs[, `:=`(start_date = as.Date(start_date), end_date = as.Date(end_date))]
usage[, usage_date := as.Date(usage_date)]
tickets[, `:=`(
  submitted_at = ymd_hms(submitted_at, quiet = TRUE),
  closed_at    = ymd_hms(closed_at,    quiet = TRUE)
)]
churn_ev[, churn_date := as.Date(churn_date)]

# ── 4. Feature Engineering ────────────────────────────────
# Snapshot date for tenure calculations
ref_date <- as.Date("2025-12-31")

## 4a. Feature usage aggregated at subscription level
usage_sub <- usage %>%
  group_by(subscription_id) %>%
  summarise(
    usage_events       = n(),
    usage_count_sum    = sum(usage_count,        na.rm = TRUE),
    usage_duration_sum = sum(usage_duration_secs, na.rm = TRUE),
    error_sum          = sum(error_count,         na.rm = TRUE),
    unique_features    = n_distinct(feature_name),
    beta_share         = mean(is_beta_feature,    na.rm = TRUE),
    .groups = "drop"
  )

## 4b. Subscription-level features aggregated to account level
sub_w_usage <- subs %>%
  left_join(usage_sub, by = "subscription_id") %>%
  mutate(
    across(c(usage_events, usage_count_sum, usage_duration_sum,
             error_sum, unique_features, beta_share), ~replace_na(., 0)),
    # Subscription tenure: open subs use ref_date as end
    sub_tenure_days = as.numeric(
      coalesce(end_date, ref_date) - start_date
    )
  )

acc_sub <- sub_w_usage %>%
  group_by(account_id) %>%
  summarise(
    subs_n              = n(),
    mrr_mean            = mean(mrr_amount,              na.rm = TRUE),
    arr_mean            = mean(arr_amount,               na.rm = TRUE),
    annual_share        = mean(billing_frequency == "annual", na.rm = TRUE),
    auto_renew_share    = mean(auto_renew_flag,          na.rm = TRUE),
    upgrade_rate        = mean(upgrade_flag,             na.rm = TRUE),
    downgrade_rate      = mean(downgrade_flag,           na.rm = TRUE),
    sub_churn_rate      = mean(churn_flag,               na.rm = TRUE),
    sub_tenure_days_avg = mean(sub_tenure_days,          na.rm = TRUE),
    usage_count_sum     = sum(usage_count_sum,           na.rm = TRUE),
    usage_duration_sum  = sum(usage_duration_sum,        na.rm = TRUE),
    error_sum           = sum(error_sum,                 na.rm = TRUE),
    unique_features     = mean(unique_features,          na.rm = TRUE),
    beta_share          = mean(beta_share,               na.rm = TRUE),
    .groups = "drop"
  )

## 4c. Support ticket signals aggregated to account level
acc_tickets <- tickets %>%
  group_by(account_id) %>%
  summarise(
    tickets_n       = n(),
    resolution_hrs  = mean(resolution_time_hours,          na.rm = TRUE),
    first_resp_mins = mean(first_response_time_minutes,    na.rm = TRUE),
    escalation_rate = mean(escalation_flag,                na.rm = TRUE),
    # sat_mean: NA = no satisfaction response recorded (replaced with 0)
    sat_mean        = mean(satisfaction_score,             na.rm = TRUE),
    high_pri_share  = mean(priority %in% c("high","urgent"), na.rm = TRUE),
    .groups = "drop"
  )

## 4d. Account tenure
accounts[, account_tenure_days := as.numeric(ref_date - signup_date)]

## 4e. Assemble account-level modeling table
model_df <- accounts %>%
  left_join(acc_sub,     by = "account_id") %>%
  left_join(acc_tickets, by = "account_id") %>%
  mutate(
    # Accounts with no ticket exposure get 0 for support metrics
    across(c(tickets_n, resolution_hrs, first_resp_mins,
             escalation_rate, sat_mean, high_pri_share), ~replace_na(., 0)),
    target_churn    = as.integer(churn_flag),
    industry        = as.factor(industry),
    referral_source = as.factor(referral_source),
    plan_tier       = as.factor(plan_tier)
  )

cat(sprintf("\nModeling table: %d accounts | Churn rate: %.1f%%\n",
            nrow(model_df), 100 * mean(model_df$target_churn)))

# ── 5. Train / Test Split (stratified 70/30) ─────────────
train_idx <- createDataPartition(model_df$target_churn, p = 0.70, list = FALSE)
train_df  <- model_df[ train_idx, ]
test_df   <- model_df[-train_idx, ]

cat(sprintf("Split: train=%d (churn=%.0f%%) | test=%d (churn=%.0f%%)\n",
            nrow(train_df), 100 * mean(train_df$target_churn),
            nrow(test_df),  100 * mean(test_df$target_churn)))

y_train <- train_df$target_churn
y_test  <- test_df$target_churn

# ── 6. Define Predictor Variable Sets ────────────────────
# Three signal families per the study's feature engineering strategy:
#   (1) Subscription lifecycle/economic signals
#   (2) Engagement/product-usage signals
#   (3) Service-friction signals
num_vars <- c(
  # Lifecycle / economic
  "seats", "account_tenure_days", "mrr_mean",
  "annual_share", "auto_renew_share",
  "upgrade_rate", "downgrade_rate", "sub_churn_rate",
  "sub_tenure_days_avg",
  # Engagement
  "usage_count_sum", "usage_duration_sum",
  "error_sum", "unique_features", "beta_share",
  # Service friction
  "tickets_n", "resolution_hrs", "first_resp_mins",
  "escalation_rate", "sat_mean", "high_pri_share"
)
cat_vars <- c("industry", "referral_source", "plan_tier")

# ── 7. MODEL 1: Logistic Regression (Interpretable Baseline) ──
cat("\n── Model 1: Logistic Regression ──\n")

lr_formula <- reformulate(c(num_vars, cat_vars), response = "target_churn")
fit_lr     <- glm(lr_formula, data = train_df, family = binomial(link = "logit"))
pred_lr    <- predict(fit_lr, newdata = test_df, type = "response")

or_table <- broom::tidy(fit_lr, conf.int = TRUE, exponentiate = TRUE) %>%
  arrange(p.value)

cat("Top 10 predictors (by p-value):\n")
print(head(or_table, 10))

# ── 8. MODEL 2: Random Forest ─────────────────────────────
cat("\n── Model 2: Random Forest ──\n")

prep_rf <- function(df) {
  df %>%
    select(all_of(c(num_vars, cat_vars)), target_churn) %>%
    mutate(
      across(all_of(num_vars), ~replace_na(as.numeric(.), 0)),
      target_churn = factor(target_churn, levels = c(0, 1), labels = c("no", "yes"))
    )
}

rf_train <- prep_rf(train_df)
rf_test  <- prep_rf(test_df)

# Upweight the minority churn class to handle class imbalance
churn_ratio <- sum(rf_train$target_churn == "no") /
               sum(rf_train$target_churn == "yes")
class_wt <- c(no = 1, yes = ceiling(churn_ratio))

fit_rf  <- randomForest(
  target_churn ~ ., data = rf_train,
  ntree      = 500,
  mtry       = floor(sqrt(length(c(num_vars, cat_vars)))),
  importance = TRUE,
  classwt    = class_wt
)
pred_rf <- predict(fit_rf, newdata = rf_test, type = "prob")[, "yes"]

rf_imp_df <- importance(fit_rf, type = 1) %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  rename(importance = MeanDecreaseAccuracy) %>%
  arrange(desc(importance))

cat("Top 10 RF features:\n"); print(head(rf_imp_df, 10))

# ── 9. MODEL 3: XGBoost (Gradient-Boosted Trees) ─────────
cat("\n── Model 3: XGBoost ──\n")

# Create dummy-encoded categoricals from the FULL dataset to ensure
# consistent column set across train / test / scoring matrices
all_dummies   <- model.matrix(~ industry + referral_source + plan_tier - 1,
                              data = model_df)
train_dummies <- all_dummies[ train_idx, ]
test_dummies  <- all_dummies[-train_idx, ]

make_X <- function(df, dum_mat) {
  num_mat <- df %>%
    select(all_of(num_vars)) %>%
    mutate(across(everything(), ~replace_na(as.numeric(.), 0))) %>%
    as.matrix()
  cbind(num_mat, dum_mat)
}

X_train <- make_X(train_df, train_dummies)
X_test  <- make_X(test_df,  test_dummies)

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

xgb_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,
  max_depth        = 4,
  min_child_weight = 5,
  subsample        = 0.80,
  colsample_bytree = 0.80,
  # Adjust for class imbalance: ratio of negatives to positives
  scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
)

fit_xgb <- xgb.train(
  params                = xgb_params,
  data                  = dtrain,
  nrounds               = 400,
  watchlist             = list(train = dtrain, test = dtest),
  early_stopping_rounds = 30,
  verbose               = 0
)
pred_xgb <- predict(fit_xgb, dtest)

cat(sprintf("Best iteration: %d\n", fit_xgb$best_iteration))

xgb_imp_df <- xgb.importance(model = fit_xgb) %>%
  as.data.frame() %>%
  arrange(desc(Gain))

cat("Top 10 XGBoost features:\n"); print(head(xgb_imp_df, 10))

# ── 10. Model Comparison: ROC / AUC ──────────────────────
cat("\n── Model Comparison (same test set) ──\n")

roc_lr  <- roc(y_test, pred_lr,  quiet = TRUE)
roc_rf  <- roc(y_test, pred_rf,  quiet = TRUE)
roc_xgb <- roc(y_test, pred_xgb, quiet = TRUE)

auc_tbl <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  AUC   = round(c(auc(roc_lr), auc(roc_rf), auc(roc_xgb)), 4)
)
print(auc_tbl)

# ROC curve plot
roc_plot_df <- bind_rows(
  data.frame(fpr = 1 - roc_lr$specificities,
             tpr = roc_lr$sensitivities,
             Model = paste0("Logistic Regression (AUC = ", round(auc(roc_lr), 3), ")")),
  data.frame(fpr = 1 - roc_rf$specificities,
             tpr = roc_rf$sensitivities,
             Model = paste0("Random Forest (AUC = ", round(auc(roc_rf), 3), ")")),
  data.frame(fpr = 1 - roc_xgb$specificities,
             tpr = roc_xgb$sensitivities,
             Model = paste0("XGBoost (AUC = ", round(auc(roc_xgb), 3), ")"))
)

p_roc <- ggplot(roc_plot_df, aes(x = fpr, y = tpr, color = Model)) +
  geom_line(linewidth = 1.1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  scale_color_manual(values = c("steelblue", "darkorange", "forestgreen")) +
  labs(
    title = "ROC Curves — Model Comparison",
    x     = "False Positive Rate (1 - Specificity)",
    y     = "True Positive Rate (Sensitivity)",
    color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(file.path(OUT_DIR, "a2_roc_comparison.png"), p_roc,
       width = 7, height = 6, dpi = 150)

# ── 11. Threshold Calibration (XGBoost — best model) ─────
cat("\n── Threshold Calibration (XGBoost) ──\n")

thresholds <- seq(0.05, 0.95, by = 0.01)
thresh_df <- lapply(thresholds, function(t) {
  cls  <- as.integer(pred_xgb >= t)
  tp   <- sum(cls == 1 & y_test == 1)
  fp   <- sum(cls == 1 & y_test == 0)
  fn   <- sum(cls == 0 & y_test == 1)
  tn   <- sum(cls == 0 & y_test == 0)
  prec <- if (tp + fp > 0) tp / (tp + fp) else NA
  rec  <- if (tp + fn > 0) tp / (tp + fn) else NA
  f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0)
            2 * prec * rec / (prec + rec) else NA
  data.frame(threshold = t, precision = prec, recall = rec,
             f1 = f1, tp = tp, fp = fp, fn = fn, tn = tn)
}) %>% bind_rows()

best_t  <- thresh_df$threshold[which.max(thresh_df$f1)]
best_f1 <- max(thresh_df$f1, na.rm = TRUE)
cat(sprintf("Optimal threshold: %.2f | F1: %.3f\n", best_t, best_f1))

# Precision-Recall curve
best_row <- thresh_df[which.max(thresh_df$f1), ]
p_pr <- ggplot(thresh_df %>% filter(!is.na(precision)), aes(x = recall, y = precision)) +
  geom_line(color = "forestgreen", linewidth = 1.1) +
  geom_point(data = best_row, aes(x = recall, y = precision),
             color = "red", size = 4, shape = 18) +
  annotate("text",
           x     = best_row$recall + 0.04,
           y     = best_row$precision,
           label = sprintf("t = %.2f\nF1 = %.3f", best_t, best_f1),
           size  = 3.5, color = "red", hjust = 0) +
  labs(
    title = "Precision-Recall Curve (XGBoost)",
    x     = "Recall",
    y     = "Precision"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(OUT_DIR, "a2_precision_recall.png"), p_pr,
       width = 7, height = 5, dpi = 150)

# ── 12. Confusion Matrix at Optimal Threshold ─────────────
pred_xgb_cls <- factor(as.integer(pred_xgb >= best_t), levels = c(0, 1))
cm <- confusionMatrix(pred_xgb_cls, factor(y_test, levels = c(0, 1)), positive = "1")

cat(sprintf("\nConfusion Matrix — XGBoost @ threshold = %.2f\n", best_t))
print(cm$table)
cat(sprintf(
  "Accuracy: %.3f | Sensitivity: %.3f | Specificity: %.3f | Precision: %.3f\n",
  cm$overall["Accuracy"], cm$byClass["Sensitivity"],
  cm$byClass["Specificity"], cm$byClass["Precision"]
))

cm_metrics_tbl <- data.frame(
  Metric = c("Accuracy", "Sensitivity (Recall)", "Specificity",
             "Precision", "F1", "Threshold"),
  Value  = round(c(cm$overall["Accuracy"], cm$byClass["Sensitivity"],
                   cm$byClass["Specificity"], cm$byClass["Precision"],
                   best_f1, best_t), 3)
)

# ── 13. Risk Tier Scoring (all 500 accounts) ──────────────
cat("\n── Risk Tier Assignment ──\n")

X_all         <- make_X(model_df, all_dummies)
pred_all_prob <- predict(fit_xgb, xgb.DMatrix(X_all))

# Three-tier intervention framework:
#   High   (≥ 0.60) → immediate Customer Success outreach
#   Medium (≥ 0.35) → proactive check-in / retention offer
#   Low    (< 0.35) → standard nurture
model_df <- model_df %>%
  mutate(
    churn_prob = pred_all_prob,
    risk_tier  = case_when(
      churn_prob >= 0.60 ~ "High",
      churn_prob >= 0.35 ~ "Medium",
      TRUE               ~ "Low"
    ),
    risk_tier = factor(risk_tier, levels = c("High", "Medium", "Low"))
  )

tier_summary <- model_df %>%
  group_by(risk_tier) %>%
  summarise(
    n_accounts     = n(),
    pct_of_base    = round(100 * n() / nrow(model_df), 1),
    actual_churns  = sum(target_churn),
    churn_rate     = round(mean(target_churn), 3),
    mrr_mean       = round(mean(mrr_mean,     na.rm = TRUE), 0),
    seats_mean     = round(mean(seats,         na.rm = TRUE), 1),
    .groups = "drop"
  )
print(tier_summary)

# Risk distribution plot
p_dist <- ggplot(model_df, aes(x = churn_prob, fill = risk_tier)) +
  geom_histogram(bins = 40, color = "white", alpha = 0.85) +
  geom_vline(xintercept = c(0.35, 0.60),
             linetype = "dashed", color = "grey30", linewidth = 0.8) +
  scale_fill_manual(values = c(High = "#d73027", Medium = "#fc8d59", Low = "#4575b4")) +
  annotate("text", x = 0.175, y = Inf, label = "Low",    vjust = 2,
           fontface = "bold", color = "#4575b4",  size = 4) +
  annotate("text", x = 0.475, y = Inf, label = "Medium", vjust = 2,
           fontface = "bold", color = "#fc8d59",  size = 4) +
  annotate("text", x = 0.775, y = Inf, label = "High",   vjust = 2,
           fontface = "bold", color = "#d73027",  size = 4) +
  labs(
    title = "Predicted Churn Risk Score Distribution (XGBoost)",
    x     = "Predicted Churn Probability",
    y     = "Number of Accounts",
    fill  = "Risk Tier"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave(file.path(OUT_DIR, "a2_risk_distribution.png"), p_dist,
       width = 8, height = 5, dpi = 150)

# ── 14. Feature Importance Plots ──────────────────────────
# XGBoost — Gain-based importance
p_xgb_imp <- xgb_imp_df %>%
  head(15) %>%
  ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "forestgreen", alpha = 0.85) +
  coord_flip() +
  labs(title = "XGBoost Feature Importance (Gain)",
       x = NULL, y = "Gain") +
  theme_minimal(base_size = 12)

ggsave(file.path(OUT_DIR, "a2_xgb_importance.png"), p_xgb_imp,
       width = 7, height = 6, dpi = 150)

# Random Forest — Mean Decrease in Accuracy
p_rf_imp <- rf_imp_df %>%
  head(15) %>%
  ggplot(aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "darkorange", alpha = 0.85) +
  coord_flip() +
  labs(title = "Random Forest Variable Importance\n(Mean Decrease in Accuracy)",
       x = NULL, y = "Mean Decrease in Accuracy") +
  theme_minimal(base_size = 12)

ggsave(file.path(OUT_DIR, "a2_rf_importance.png"), p_rf_imp,
       width = 7, height = 6, dpi = 150)

# Logistic Regression — Odds Ratios (p < 0.20, non-intercept terms)
p_or <- or_table %>%
  filter(term != "(Intercept)", p.value < 0.20) %>%
  ggplot(aes(x = reorder(term, estimate), y = estimate)) +
  geom_col(aes(fill = estimate > 1), alpha = 0.85, show.legend = FALSE) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.3) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
  scale_fill_manual(values = c("TRUE" = "#d73027", "FALSE" = "#4575b4")) +
  coord_flip() +
  labs(title = "Logistic Regression Odds Ratios (p < 0.20)",
       x = NULL, y = "Odds Ratio (95% CI)") +
  theme_minimal(base_size = 12)

ggsave(file.path(OUT_DIR, "a2_logit_odds_ratios.png"), p_or,
       width = 7, height = 5, dpi = 150)

# ── 15. Save All CSV Outputs ──────────────────────────────
cat("\n── Saving outputs to:", OUT_DIR, "──\n")

# Model comparison
write.csv(auc_tbl,       file.path(OUT_DIR, "a2_auc_comparison.csv"),     row.names = FALSE)
write.csv(cm_metrics_tbl,file.path(OUT_DIR, "a2_xgb_cm_metrics.csv"),     row.names = FALSE)
write.csv(thresh_df,     file.path(OUT_DIR, "a2_threshold_metrics.csv"),  row.names = FALSE)

# Model interpretation
write.csv(or_table,      file.path(OUT_DIR, "a2_logit_odds_ratios.csv"),  row.names = FALSE)
write.csv(rf_imp_df,     file.path(OUT_DIR, "a2_rf_importance.csv"),      row.names = FALSE)
write.csv(xgb_imp_df,    file.path(OUT_DIR, "a2_xgb_importance.csv"),     row.names = FALSE)

# Retention decision-support artifacts
write.csv(tier_summary,  file.path(OUT_DIR, "a2_risk_tier_summary.csv"),  row.names = FALSE)

risk_export <- model_df %>%
  select(account_id, account_name, industry, country, plan_tier,
         seats, mrr_mean, account_tenure_days,
         churn_prob, risk_tier, target_churn) %>%
  arrange(desc(churn_prob))
write.csv(risk_export,   file.path(OUT_DIR, "a2_account_risk_scores.csv"), row.names = FALSE)

# ── 16. Final Summary ─────────────────────────────────────
cat("\n=== FINAL SUMMARY ===\n")
cat("\nAUC Comparison:\n"); print(auc_tbl)
cat(sprintf("\nOptimal XGBoost threshold: %.2f | F1: %.3f\n", best_t, best_f1))
cat("\nRisk Tier Breakdown:\n"); print(tier_summary)
cat("\nCSVs saved:\n")
cat("  a2_auc_comparison.csv\n  a2_xgb_cm_metrics.csv\n")
cat("  a2_threshold_metrics.csv\n  a2_logit_odds_ratios.csv\n")
cat("  a2_rf_importance.csv\n  a2_xgb_importance.csv\n")
cat("  a2_risk_tier_summary.csv\n  a2_account_risk_scores.csv\n")
cat("\nPlots saved:\n")
cat("  a2_roc_comparison.png\n  a2_precision_recall.png\n")
cat("  a2_risk_distribution.png\n  a2_xgb_importance.png\n")
cat("  a2_rf_importance.png\n  a2_logit_odds_ratios.png\n")
cat("\nDone!\n")
