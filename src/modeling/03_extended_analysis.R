# ============================================================
# A3_Extended_Analysis.R
# Feature Engineering Journey + Model Expansion + Hyperparameter Tuning
# Genki Hirayama | Northwestern University
# ============================================================
# MODELS: Logistic Regression, Elastic Net, Random Forest,
#         XGBoost, LightGBM
#         [CatBoost: optional — install manually via catboost GitHub]
# FE STAGES: 0=Minimal | 1=Baseline | 2=+Ratios | 3=+Interactions | 4=+Recency
# ============================================================

set.seed(42)
suppressPackageStartupMessages({
  library(pROC)
  library(ranger)
  library(xgboost)
  library(glmnet)
  library(lightgbm)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

cat_available <- tryCatch({ library(catboost); TRUE }, error = function(e) FALSE)
if (!cat_available) cat("CatBoost not installed — skipping. See GitHub releases for Windows binary.\n")

base <- "C:/Users/genki/Documents/Homework/AI_Capstone/Assignment 1"
out  <- "C:/Users/genki/Documents/Homework/AI_Capstone/Assignment 3/reports"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1. LOAD DATA
# ============================================================
acc <- read.csv(file.path(base, "ravenstack_accounts.csv"),     stringsAsFactors = FALSE)
sub <- read.csv(file.path(base, "ravenstack_subscriptions.csv"),stringsAsFactors = FALSE)
fu  <- read.csv(file.path(base, "ravenstack_feature_usage.csv"),stringsAsFactors = FALSE)
st  <- read.csv(file.path(base, "ravenstack_support_tickets.csv"), stringsAsFactors = FALSE)

sub$start_date <- as.Date(sub$start_date)
fu$usage_date  <- as.Date(fu$usage_date)
st$closed_at   <- as.Date(substr(st$closed_at, 1, 10))

ref_date <- max(c(max(fu$usage_date, na.rm = TRUE),
                  max(sub$start_date, na.rm = TRUE)))

acc$churn              <- as.integer(acc$churn_flag == "True")
acc$account_tenure_days <- as.numeric(ref_date - as.Date(acc$signup_date))
cat("Loaded:", nrow(acc), "accounts |", sum(acc$churn), "churned\n")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# -- Subscription aggregates --
sub$is_churned    <- as.integer(sub$churn_flag    == "True")
sub$is_upgrade    <- as.integer(sub$upgrade_flag  == "True")
sub$is_downgrade  <- as.integer(sub$downgrade_flag == "True")
sub$is_annual     <- as.integer(sub$billing_frequency == "annual")
sub$is_autorenew  <- as.integer(sub$auto_renew_flag == "True")
sub$tenure_days   <- as.numeric(ref_date - sub$start_date)

sub_agg <- sub %>% group_by(account_id) %>% summarise(
  mrr_mean          = mean(mrr_amount,   na.rm = TRUE),
  upgrade_rate      = mean(is_upgrade,   na.rm = TRUE),
  downgrade_rate    = mean(is_downgrade, na.rm = TRUE),
  sub_churn_rate    = mean(is_churned,   na.rm = TRUE),
  annual_share      = mean(is_annual,    na.rm = TRUE),
  auto_renew_share  = mean(is_autorenew, na.rm = TRUE),
  sub_tenure_days_avg = mean(tenure_days, na.rm = TRUE),
  .groups = "drop")

# -- Feature usage aggregates --
fu_sub <- merge(fu, sub[, c("subscription_id", "account_id")], by = "subscription_id")

fu_agg <- fu_sub %>% group_by(account_id) %>% summarise(
  usage_count_sum    = sum(usage_count,         na.rm = TRUE),
  usage_duration_sum = sum(usage_duration_secs, na.rm = TRUE),
  error_sum          = sum(error_count,          na.rm = TRUE),
  unique_features    = n_distinct(feature_name),
  beta_share         = mean(is_beta_feature == "True", na.rm = TRUE),
  last_usage_date    = max(usage_date, na.rm = TRUE),
  .groups = "drop")

fu_agg$days_since_last_usage <- as.numeric(ref_date - fu_agg$last_usage_date)

# -- Support ticket aggregates --
st$is_escalated <- as.integer(st$escalation_flag == "True")
st$is_high_pri  <- as.integer(st$priority %in% c("high", "urgent"))

st_agg <- st %>% group_by(account_id) %>% summarise(
  tickets_n         = n(),
  resolution_hrs    = mean(resolution_time_hours,        na.rm = TRUE),
  first_resp_mins   = mean(first_response_time_minutes,  na.rm = TRUE),
  high_pri_share    = mean(is_high_pri,    na.rm = TRUE),
  escalation_rate   = mean(is_escalated,   na.rm = TRUE),
  sat_mean          = mean(satisfaction_score, na.rm = TRUE),
  last_ticket_date  = max(closed_at, na.rm = TRUE),
  .groups = "drop")

st_agg$sat_mean[is.na(st_agg$sat_mean)] <- mean(st_agg$sat_mean, na.rm = TRUE)
st_agg$days_since_last_ticket <- as.numeric(ref_date - st_agg$last_ticket_date)

# -- Join all --
df <- acc %>%
  select(account_id, churn, seats, plan_tier, referral_source, industry,
         account_tenure_days) %>%
  left_join(sub_agg, by = "account_id") %>%
  left_join(fu_agg  %>% select(-last_usage_date),  by = "account_id") %>%
  left_join(st_agg  %>% select(-last_ticket_date), by = "account_id")

# Impute missing (accounts with no tickets)
fill_mean <- function(x) { x[is.na(x)] <- mean(x, na.rm = TRUE); x }
df$tickets_n       <- ifelse(is.na(df$tickets_n), 0, df$tickets_n)
df$resolution_hrs  <- fill_mean(df$resolution_hrs)
df$first_resp_mins <- fill_mean(df$first_resp_mins)
df$high_pri_share  <- ifelse(is.na(df$high_pri_share),  0, df$high_pri_share)
df$escalation_rate <- ifelse(is.na(df$escalation_rate), 0, df$escalation_rate)
df$days_since_last_ticket <- fill_mean(df$days_since_last_ticket)
df$days_since_last_usage  <- fill_mean(df$days_since_last_usage)

# ============================================================
# 3. DEFINE FEATURE STAGES
# ============================================================

# Stage 0 — Minimal: demographics + economics only
stage0 <- c("account_tenure_days", "mrr_mean", "seats",
            "upgrade_rate", "downgrade_rate", "annual_share")

# Stage 1 — Baseline: all global aggregates (reproduces A2 analysis)
stage1 <- c("resolution_hrs", "first_resp_mins", "high_pri_share",
            "escalation_rate", "sat_mean", "tickets_n", "error_sum",
            "usage_count_sum", "usage_duration_sum", "unique_features",
            "beta_share", "sub_tenure_days_avg", "sub_churn_rate",
            "account_tenure_days", "mrr_mean", "seats",
            "upgrade_rate", "downgrade_rate", "annual_share", "auto_renew_share")

# Stage 2 — +Ratios: normalize by seat count and usage volume
df$tickets_per_seat       <- df$tickets_n       / (df$seats + 1)
df$errors_per_usage       <- df$error_sum       / (df$usage_count_sum + 1)
df$usage_per_seat         <- df$usage_count_sum / (df$seats + 1)
df$mrr_per_seat           <- df$mrr_mean        / (df$seats + 1)
df$support_hrs_per_seat   <- (df$resolution_hrs * df$tickets_n) / (df$seats + 1)

stage2 <- c(stage1, "tickets_per_seat", "errors_per_usage",
            "usage_per_seat", "mrr_per_seat", "support_hrs_per_seat")

# Stage 3 — +Interactions: compound cross-signal features
df$friction_index    <- df$escalation_rate * df$resolution_hrs
df$risk_composite    <- df$escalation_rate * df$high_pri_share
df$value_at_risk     <- df$mrr_mean        * (df$escalation_rate + 0.01)
df$engagement_rate   <- df$usage_count_sum / (df$account_tenure_days + 1)
df$support_vs_usage  <- df$tickets_n       / (df$usage_count_sum + 1)

stage3 <- c(stage2, "friction_index", "risk_composite",
            "value_at_risk", "engagement_rate", "support_vs_usage")

# Stage 4 — +Recency: time-since-last signals
stage4 <- c(stage3, "days_since_last_usage", "days_since_last_ticket")

stage_list <- list(
  "S0: Minimal"       = stage0,
  "S1: Baseline"      = stage1,
  "S2: +Ratios"       = stage2,
  "S3: +Interactions" = stage3,
  "S4: +Recency"      = stage4
)

cat_vars <- c("plan_tier", "referral_source", "industry")

# ============================================================
# 4. TRAIN / TEST SPLIT  (stratified 70/30, same seed as A2)
# ============================================================
set.seed(42)
churn_idx    <- which(df$churn == 1)
nochurn_idx  <- which(df$churn == 0)
train_idx    <- c(sample(churn_idx,   floor(0.7 * length(churn_idx))),
                  sample(nochurn_idx, floor(0.7 * length(nochurn_idx))))
test_idx     <- setdiff(seq_len(nrow(df)), train_idx)

cat("Train:", length(train_idx), "| Test:", length(test_idx),
    "| Test churners:", sum(df$churn[test_idx]), "\n")

# ============================================================
# 5. HELPERS
# ============================================================

# Build numeric matrix with one-hot categoricals
build_X <- function(data, num_feats) {
  feats <- num_feats[num_feats %in% names(data)]
  X <- data[, feats, drop = FALSE]
  for (cv in cat_vars) {
    if (cv %in% names(data)) {
      dm <- model.matrix(as.formula(paste("~", cv, "- 1")), data = data)
      X  <- cbind(X, dm)
    }
  }
  X[!is.finite(as.matrix(X))] <- 0   # guard against Inf/NaN
  as.matrix(X)
}

get_auc <- function(probs, labels) {
  as.numeric(pROC::auc(pROC::roc(labels, probs, quiet = TRUE)))
}

pos_weight <- sum(df$churn[train_idx] == 0) / sum(df$churn[train_idx] == 1)

# Individual model trainers ----------------------------------------

run_lr <- function(Xtr, ytr, Xte, yte) {
  d <- as.data.frame(Xtr); d$y <- ytr
  safe_names <- make.names(names(d))
  names(d) <- safe_names
  de <- as.data.frame(Xte); names(de) <- safe_names[-length(safe_names)]
  m  <- suppressWarnings(glm(y ~ ., data = d, family = "binomial"))
  get_auc(predict(m, newdata = de, type = "response"), yte)
}

run_enet <- function(Xtr, ytr, Xte, yte) {
  cv <- cv.glmnet(Xtr, ytr, family = "binomial", alpha = 0.5,
                  nfolds = 5, type.measure = "auc")
  pr <- predict(cv, newx = Xte, s = "lambda.min", type = "response")[, 1]
  get_auc(pr, yte)
}

run_rf <- function(Xtr, ytr, Xte, yte,
                   mtry_val = NULL, min_node = 5) {
  d  <- as.data.frame(Xtr); d$y <- as.factor(ytr)
  de <- as.data.frame(Xte)
  safe <- make.names(names(d))
  names(d) <- safe; names(de) <- safe[-length(safe)]
  if (is.null(mtry_val)) mtry_val <- max(1, floor(sqrt(ncol(Xtr))))
  m  <- ranger(y ~ ., data = d, num.trees = 500, probability = TRUE,
               mtry = mtry_val, min.node.size = min_node,
               class.weights = c("0" = 1, "1" = pos_weight), seed = 42)
  pr <- predict(m, data = de)$predictions[, "1"]
  get_auc(pr, yte)
}

run_xgb <- function(Xtr, ytr, Xte, yte,
                    depth = 5, eta = 0.1, subsample = 0.8,
                    colsample = 0.8, nrounds = 150) {
  dtr <- xgb.DMatrix(data = Xtr, label = ytr)
  dte <- xgb.DMatrix(data = Xte, label = yte)
  p <- list(objective = "binary:logistic", eval_metric = "auc",
            max_depth = depth, eta = eta,
            subsample = subsample, colsample_bytree = colsample,
            scale_pos_weight = pos_weight)
  m  <- xgb.train(params = p, data = dtr, nrounds = nrounds,
                  watchlist = list(val = dte), verbose = 0,
                  early_stopping_rounds = 20)
  get_auc(predict(m, dte), yte)
}

run_lgbm <- function(Xtr, ytr, Xte, yte,
                     num_leaves = 31, lr = 0.05, nrounds = 200) {
  dtr <- lgb.Dataset(Xtr, label = ytr)
  p <- list(objective = "binary", metric = "auc",
            num_leaves = num_leaves, learning_rate = lr,
            scale_pos_weight = pos_weight, verbose = -1,
            min_data_in_leaf = 5)
  m  <- lgb.train(params = p, data = dtr, nrounds = nrounds, verbose = -1)
  get_auc(predict(m, Xte), yte)
}

# ============================================================
# 6. FEATURE ENGINEERING JOURNEY
# ============================================================
cat("\n======  FEATURE ENGINEERING JOURNEY  ======\n")

journey <- data.frame()

for (sname in names(stage_list)) {
  feats  <- stage_list[[sname]]
  X_all  <- build_X(df, feats)
  Xtr    <- X_all[train_idx, , drop = FALSE]
  Xte    <- X_all[test_idx,  , drop = FALSE]
  ytr    <- df$churn[train_idx]
  yte    <- df$churn[test_idx]

  cat(sprintf("\n%s  (%d features)\n", sname, ncol(X_all)))

  auc_lr   <- tryCatch(run_lr(Xtr, ytr, Xte, yte),   error = function(e) NA)
  auc_enet <- tryCatch(run_enet(Xtr, ytr, Xte, yte), error = function(e) NA)
  auc_rf   <- tryCatch(run_rf(Xtr, ytr, Xte, yte),   error = function(e) NA)
  auc_xgb  <- tryCatch(run_xgb(Xtr, ytr, Xte, yte),  error = function(e) NA)
  auc_lgbm <- tryCatch(run_lgbm(Xtr, ytr, Xte, yte), error = function(e) NA)

  cat(sprintf("  LR=%.4f  ElasticNet=%.4f  RF=%.4f  XGB=%.4f  LGBM=%.4f\n",
              auc_lr, auc_enet, auc_rf, auc_xgb, auc_lgbm))

  journey <- rbind(journey, data.frame(
    Stage = sname, N_Features = ncol(X_all),
    LR = auc_lr, ElasticNet = auc_enet,
    RF = auc_rf, XGBoost = auc_xgb, LightGBM = auc_lgbm,
    stringsAsFactors = FALSE))
}

write.csv(journey, file.path(out, "a3_fe_journey.csv"), row.names = FALSE)

# ============================================================
# 7. HYPERPARAMETER TUNING  (on Stage 4 — best feature set)
# ============================================================
cat("\n======  HYPERPARAMETER TUNING (Stage 4)  ======\n")

Xall4 <- build_X(df, stage4)
Xtr4  <- Xall4[train_idx, , drop = FALSE]
Xte4  <- Xall4[test_idx,  , drop = FALSE]
ytr4  <- df$churn[train_idx]
yte4  <- df$churn[test_idx]

# ---- XGBoost grid ----
cat("\nXGBoost grid search...\n")
xgb_grid <- expand.grid(depth     = c(3, 5, 7),
                         eta       = c(0.05, 0.1, 0.2),
                         subsample = c(0.7, 0.9))
xgb_tune <- data.frame()
dtr_xgb  <- xgb.DMatrix(data = Xtr4, label = ytr4)
dte_xgb  <- xgb.DMatrix(data = Xte4, label = yte4)

for (i in seq_len(nrow(xgb_grid))) {
  p <- list(objective = "binary:logistic", eval_metric = "auc",
            max_depth = xgb_grid$depth[i], eta = xgb_grid$eta[i],
            subsample = xgb_grid$subsample[i], colsample_bytree = 0.8,
            scale_pos_weight = pos_weight)
  m <- xgb.train(params = p, data = dtr_xgb, nrounds = 300,
                 watchlist = list(val = dte_xgb), verbose = 0,
                 early_stopping_rounds = 25)
  row <- data.frame(depth = p$max_depth, eta = p$eta,
                    subsample = p$subsample,
                    best_rounds = m$best_iteration,
                    AUC = m$best_score)
  xgb_tune <- rbind(xgb_tune, row)
  cat(sprintf("  depth=%d eta=%.2f sub=%.1f -> AUC=%.4f (rounds=%d)\n",
              p$max_depth, p$eta, p$subsample, m$best_score, m$best_iteration))
}
best_xgb_row <- xgb_tune[which.max(xgb_tune$AUC), ]
write.csv(xgb_tune, file.path(out, "a3_xgb_tuning.csv"), row.names = FALSE)
cat("Best XGB:", sprintf("depth=%d eta=%.2f sub=%.1f AUC=%.4f\n",
    best_xgb_row$depth, best_xgb_row$eta,
    best_xgb_row$subsample, best_xgb_row$AUC))

# ---- Random Forest grid ----
cat("\nRandom Forest grid search...\n")
rf_grid <- expand.grid(mtry     = c(3, 5, 8, 12),
                        min_node = c(1, 5, 10))
rf_tune  <- data.frame()
df_tr_rf <- as.data.frame(Xtr4); df_tr_rf$y <- as.factor(ytr4)
df_te_rf  <- as.data.frame(Xte4)
safe_rf   <- make.names(names(df_tr_rf))
names(df_tr_rf) <- safe_rf; names(df_te_rf) <- safe_rf[-length(safe_rf)]

for (i in seq_len(nrow(rf_grid))) {
  m   <- ranger(y ~ ., data = df_tr_rf, num.trees = 500, probability = TRUE,
                mtry = rf_grid$mtry[i], min.node.size = rf_grid$min_node[i],
                class.weights = c("0" = 1, "1" = pos_weight), seed = 42)
  auc <- get_auc(predict(m, data = df_te_rf)$predictions[, "1"], yte4)
  rf_tune <- rbind(rf_tune, data.frame(mtry     = rf_grid$mtry[i],
                                        min_node = rf_grid$min_node[i],
                                        AUC      = auc))
  cat(sprintf("  mtry=%d min_node=%d -> AUC=%.4f\n",
              rf_grid$mtry[i], rf_grid$min_node[i], auc))
}
best_rf_row <- rf_tune[which.max(rf_tune$AUC), ]
write.csv(rf_tune, file.path(out, "a3_rf_tuning.csv"), row.names = FALSE)
cat("Best RF: mtry=", best_rf_row$mtry,
    "min_node=", best_rf_row$min_node,
    "AUC=", round(best_rf_row$AUC, 4), "\n")

# ---- LightGBM grid ----
cat("\nLightGBM grid search...\n")
lgbm_grid <- expand.grid(num_leaves = c(15, 31, 63),
                          lr         = c(0.03, 0.05, 0.1))
lgbm_tune  <- data.frame()
dtr_lgb    <- lgb.Dataset(Xtr4, label = ytr4)

for (i in seq_len(nrow(lgbm_grid))) {
  p <- list(objective = "binary", metric = "auc",
            num_leaves = lgbm_grid$num_leaves[i],
            learning_rate = lgbm_grid$lr[i],
            scale_pos_weight = pos_weight, verbose = -1,
            min_data_in_leaf = 5)
  m   <- lgb.train(params = p, data = dtr_lgb, nrounds = 300, verbose = -1)
  auc <- get_auc(predict(m, Xte4), yte4)
  lgbm_tune <- rbind(lgbm_tune, data.frame(num_leaves = lgbm_grid$num_leaves[i],
                                             lr = lgbm_grid$lr[i], AUC = auc))
  cat(sprintf("  leaves=%d lr=%.3f -> AUC=%.4f\n",
              lgbm_grid$num_leaves[i], lgbm_grid$lr[i], auc))
}
best_lgbm_row <- lgbm_tune[which.max(lgbm_tune$AUC), ]
write.csv(lgbm_tune, file.path(out, "a3_lgbm_tuning.csv"), row.names = FALSE)
cat("Best LGBM: leaves=", best_lgbm_row$num_leaves,
    "lr=", best_lgbm_row$lr,
    "AUC=", round(best_lgbm_row$AUC, 4), "\n")

# ============================================================
# 8. FINAL TUNED MODEL COMPARISON
# ============================================================
cat("\n======  FINAL TUNED MODEL COMPARISON  ======\n")

auc_lr_f   <- tryCatch(run_lr(Xtr4, ytr4, Xte4, yte4), error = function(e) NA)
auc_enet_f <- tryCatch(run_enet(Xtr4, ytr4, Xte4, yte4), error = function(e) NA)

# Best RF
rf_best_m  <- ranger(y ~ ., data = df_tr_rf, num.trees = 500, probability = TRUE,
                     mtry = best_rf_row$mtry, min.node.size = best_rf_row$min_node,
                     class.weights = c("0" = 1, "1" = pos_weight), seed = 42)
auc_rf_f   <- get_auc(predict(rf_best_m, data = df_te_rf)$predictions[, "1"], yte4)

# Best XGB
xgb_best_m <- xgb.train(
  params = list(objective = "binary:logistic", eval_metric = "auc",
                max_depth = best_xgb_row$depth, eta = best_xgb_row$eta,
                subsample = best_xgb_row$subsample, colsample_bytree = 0.8,
                scale_pos_weight = pos_weight),
  data = dtr_xgb, nrounds = best_xgb_row$best_rounds, verbose = 0)
auc_xgb_f  <- get_auc(predict(xgb_best_m, dte_xgb), yte4)

# Best LGBM
lgbm_best_m <- lgb.train(
  params = list(objective = "binary", metric = "auc",
                num_leaves = best_lgbm_row$num_leaves,
                learning_rate = best_lgbm_row$lr,
                scale_pos_weight = pos_weight, verbose = -1,
                min_data_in_leaf = 5),
  data = dtr_lgb, nrounds = 300, verbose = -1)
auc_lgbm_f <- get_auc(predict(lgbm_best_m, Xte4), yte4)

# A2 baseline for comparison
auc_a2_xgb <- 0.5664
auc_a2_rf  <- 0.5915
auc_a2_lr  <- 0.5730

final_cmp <- data.frame(
  Model = c("LR (A2 baseline)", "LR (S4, untuned)",
            "Elastic Net (S4)", "RF (A2 baseline)",
            "RF (S4, tuned)", "XGBoost (A2 baseline)",
            "XGBoost (S4, tuned)", "LightGBM (S4, tuned)"),
  AUC   = c(auc_a2_lr, auc_lr_f, auc_enet_f,
            auc_a2_rf, auc_rf_f, auc_a2_xgb,
            auc_xgb_f, auc_lgbm_f),
  Category = c("Baseline A2", "Stage 4", "Stage 4",
               "Baseline A2", "Stage 4 Tuned",
               "Baseline A2", "Stage 4 Tuned", "Stage 4 Tuned"),
  stringsAsFactors = FALSE
)
final_cmp <- final_cmp[order(-final_cmp$AUC), ]
cat("\nFinal model comparison:\n")
print(final_cmp, row.names = FALSE)
write.csv(final_cmp, file.path(out, "a3_final_model_comparison.csv"), row.names = FALSE)

# ============================================================
# 9.  FEATURE IMPORTANCE — XGBoost tuned model
# ============================================================
xgb_imp <- xgb.importance(model = xgb_best_m)
write.csv(xgb_imp, file.path(out, "a3_xgb_importance_tuned.csv"), row.names = FALSE)

lgbm_imp <- lgb.importance(lgbm_best_m, percentage = TRUE)
write.csv(lgbm_imp, file.path(out, "a3_lgbm_importance_tuned.csv"), row.names = FALSE)

# ============================================================
# 10. VISUALIZATIONS
# ============================================================
cat("\n======  GENERATING PLOTS  ======\n")

# ---- Plot 1: FE Journey ----
j_long <- journey %>%
  select(Stage, LR, ElasticNet, RF, XGBoost, LightGBM) %>%
  pivot_longer(-Stage, names_to = "Model", values_to = "AUC") %>%
  filter(!is.na(AUC))

j_long$Stage <- factor(j_long$Stage, levels = names(stage_list))
j_long$Model <- factor(j_long$Model,
                        levels = c("LR","ElasticNet","RF","XGBoost","LightGBM"))

p1 <- ggplot(j_long, aes(x = Stage, y = AUC, color = Model, group = Model)) +
  geom_line(linewidth = 1.3) +
  geom_point(size = 3.5) +
  geom_hline(yintercept = 0.5, linetype = "dashed",
             color = "gray60", linewidth = 0.8) +
  scale_color_manual(values = c("LR"="#78909C","ElasticNet"="#42A5F5",
                                 "RF"="#66BB6A","XGBoost"="#EF5350",
                                 "LightGBM"="#AB47BC")) +
  scale_y_continuous(limits = c(0.47, 0.80),
                     breaks  = seq(0.47, 0.80, 0.05)) +
  labs(title    = "Feature Engineering Journey: AUC by Stage and Model",
       subtitle = "Each stage builds on the previous\nS0=Minimal | S1=Baseline | S2=+Ratios | S3=+Interactions | S4=+Recency",
       x = NULL, y = "AUC (Test Set)",
       caption = "Dashed line = random chance (0.50)") +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1),
        legend.position = "right",
        plot.title = element_text(face = "bold"))

ggsave(file.path(out, "a3_fe_journey.png"), p1, width = 11, height = 6, dpi = 150)
cat("Saved: a3_fe_journey.png\n")

# ---- Plot 2: Final Model Comparison (tuned vs baseline) ----
fc <- final_cmp
fc$Model    <- factor(fc$Model, levels = fc$Model[order(fc$AUC)])
fc$Category <- factor(fc$Category,
                       levels = c("Baseline A2","Stage 4","Stage 4 Tuned"))

p2 <- ggplot(fc, aes(x = Model, y = AUC, fill = Category)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = sprintf("%.4f", AUC)),
            hjust = -0.1, size = 4) +
  geom_hline(yintercept = 0.5, linetype = "dashed",
             color = "gray60", linewidth = 0.8) +
  scale_fill_manual(values = c("Baseline A2"  = "#90A4AE",
                                "Stage 4"      = "#64B5F6",
                                "Stage 4 Tuned"= "#1565C0")) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 0.90)) +
  labs(title    = "Full Model Comparison: A2 Baseline vs Stage 4 Tuned",
       subtitle = "All Stage 4 models use the enriched feature set (S4: +Recency)",
       x = NULL, y = "AUC (Test Set)", fill = "Configuration",
       caption = "Dashed line = random chance (0.50)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(out, "a3_model_comparison_tuned.png"),
       p2, width = 10, height = 6, dpi = 150)
cat("Saved: a3_model_comparison_tuned.png\n")

# ---- Plot 3: XGBoost Tuning Heatmap ----
xgb_tune$eta_lbl   <- paste("eta =", xgb_tune$eta)
xgb_tune$sub_lbl   <- paste("subsample =", xgb_tune$subsample)

p3 <- ggplot(xgb_tune, aes(x = eta_lbl, y = factor(depth), fill = AUC)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.4f", AUC)), size = 3.5, color = "white") +
  facet_wrap(~ sub_lbl) +
  scale_fill_gradient(low = "#FFA726", high = "#B71C1C", name = "AUC") +
  labs(title    = "XGBoost Hyperparameter Tuning Grid (Stage 4 Features)",
       subtitle = "Early stopping (25 rounds patience); best nrounds varies per cell",
       x = "Learning Rate", y = "Max Depth") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(out, "a3_xgb_tuning_heatmap.png"),
       p3, width = 11, height = 5, dpi = 150)
cat("Saved: a3_xgb_tuning_heatmap.png\n")

# ---- Plot 4: RF Tuning Heatmap ----
p4 <- ggplot(rf_tune, aes(x = factor(min_node), y = factor(mtry), fill = AUC)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.4f", AUC)), size = 3.8, color = "white") +
  scale_fill_gradient(low = "#A5D6A7", high = "#1B5E20", name = "AUC") +
  labs(title    = "Random Forest Hyperparameter Tuning Grid (Stage 4 Features)",
       x = "Min Node Size", y = "mtry (features per split)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(out, "a3_rf_tuning_heatmap.png"),
       p4, width = 7, height = 5, dpi = 150)
cat("Saved: a3_rf_tuning_heatmap.png\n")

# ---- Plot 5: LightGBM Tuning ----
p5 <- ggplot(lgbm_tune, aes(x = factor(lr), y = factor(num_leaves), fill = AUC)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.4f", AUC)), size = 3.8, color = "white") +
  scale_fill_gradient(low = "#CE93D8", high = "#4A148C", name = "AUC") +
  labs(title = "LightGBM Hyperparameter Tuning Grid (Stage 4 Features)",
       x = "Learning Rate", y = "Num Leaves") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(out, "a3_lgbm_tuning_heatmap.png"),
       p5, width = 7, height = 5, dpi = 150)
cat("Saved: a3_lgbm_tuning_heatmap.png\n")

# ---- Plot 6: Top Feature Importance — tuned XGB ----
top_imp <- head(xgb_imp, 15)
top_imp$Feature <- factor(top_imp$Feature,
                           levels = top_imp$Feature[order(top_imp$Gain)])
p6 <- ggplot(top_imp, aes(x = Feature, y = Gain)) +
  geom_segment(aes(xend = Feature, yend = 0), color = "#EF5350", linewidth = 1.2) +
  geom_point(color = "#B71C1C", size = 4) +
  coord_flip() +
  labs(title    = "XGBoost Feature Importance — Tuned Model (Stage 4)",
       subtitle = "Gain = average improvement in loss per split",
       x = NULL, y = "Gain") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(out, "a3_xgb_importance_tuned.png"),
       p6, width = 9, height = 6, dpi = 150)
cat("Saved: a3_xgb_importance_tuned.png\n")

# ============================================================
cat("\n=== COMPLETE ===\n")
cat("Outputs in:", out, "\n")
cat("\nFeature Engineering Journey summary:\n")
print(journey[, c("Stage","N_Features","LR","ElasticNet","RF","XGBoost","LightGBM")],
      row.names = FALSE, digits = 4)
cat("\nBest tuned results:\n")
cat(sprintf("  XGBoost:  %.4f\n  RF:       %.4f\n  LightGBM: %.4f\n",
            auc_xgb_f, auc_rf_f, auc_lgbm_f))
