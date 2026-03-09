# Fix script: re-run final model comparison with corrected XGBoost training
suppressPackageStartupMessages({
  library(pROC); library(ranger); library(xgboost)
  library(glmnet); library(lightgbm); library(dplyr); library(ggplot2)
})

base <- "C:/Users/genki/Documents/Homework/AI_Capstone/Assignment 1"
out  <- "C:/Users/genki/Documents/Homework/AI_Capstone/Assignment 3/reports"

# Re-load tuning results
xgb_tune  <- read.csv(file.path(out, "a3_xgb_tuning.csv"))
rf_tune   <- read.csv(file.path(out, "a3_rf_tuning.csv"))
lgbm_tune <- read.csv(file.path(out, "a3_lgbm_tuning.csv"))

# Reload data (same pipeline as main script)
set.seed(42)
acc <- read.csv(file.path(base, "ravenstack_accounts.csv"), stringsAsFactors=FALSE)
sub <- read.csv(file.path(base, "ravenstack_subscriptions.csv"), stringsAsFactors=FALSE)
fu  <- read.csv(file.path(base, "ravenstack_feature_usage.csv"), stringsAsFactors=FALSE)
st  <- read.csv(file.path(base, "ravenstack_support_tickets.csv"), stringsAsFactors=FALSE)

sub$start_date <- as.Date(sub$start_date)
fu$usage_date  <- as.Date(fu$usage_date)
st$closed_at   <- as.Date(substr(st$closed_at, 1, 10))
ref_date <- max(c(max(fu$usage_date, na.rm=TRUE), max(sub$start_date, na.rm=TRUE)))

acc$churn <- as.integer(acc$churn_flag == "True")
acc$account_tenure_days <- as.numeric(ref_date - as.Date(acc$signup_date))

sub$is_churned   <- as.integer(sub$churn_flag    == "True")
sub$is_upgrade   <- as.integer(sub$upgrade_flag  == "True")
sub$is_downgrade <- as.integer(sub$downgrade_flag == "True")
sub$is_annual    <- as.integer(sub$billing_frequency == "annual")
sub$is_autorenew <- as.integer(sub$auto_renew_flag == "True")
sub$tenure_days  <- as.numeric(ref_date - sub$start_date)

sub_agg <- sub %>% group_by(account_id) %>% summarise(
  mrr_mean=mean(mrr_amount,na.rm=TRUE), upgrade_rate=mean(is_upgrade,na.rm=TRUE),
  downgrade_rate=mean(is_downgrade,na.rm=TRUE), sub_churn_rate=mean(is_churned,na.rm=TRUE),
  annual_share=mean(is_annual,na.rm=TRUE), auto_renew_share=mean(is_autorenew,na.rm=TRUE),
  sub_tenure_days_avg=mean(tenure_days,na.rm=TRUE), .groups="drop")

fu_sub <- merge(fu, sub[,c("subscription_id","account_id")], by="subscription_id")
fu_agg <- fu_sub %>% group_by(account_id) %>% summarise(
  usage_count_sum=sum(usage_count,na.rm=TRUE),
  usage_duration_sum=sum(usage_duration_secs,na.rm=TRUE),
  error_sum=sum(error_count,na.rm=TRUE), unique_features=n_distinct(feature_name),
  beta_share=mean(is_beta_feature=="True",na.rm=TRUE),
  last_usage_date=max(usage_date,na.rm=TRUE), .groups="drop")
fu_agg$days_since_last_usage <- as.numeric(ref_date - fu_agg$last_usage_date)

st$is_escalated <- as.integer(st$escalation_flag=="True")
st$is_high_pri  <- as.integer(st$priority %in% c("high","urgent"))
st_agg <- st %>% group_by(account_id) %>% summarise(
  tickets_n=n(), resolution_hrs=mean(resolution_time_hours,na.rm=TRUE),
  first_resp_mins=mean(first_response_time_minutes,na.rm=TRUE),
  high_pri_share=mean(is_high_pri,na.rm=TRUE),
  escalation_rate=mean(is_escalated,na.rm=TRUE),
  sat_mean=mean(satisfaction_score,na.rm=TRUE),
  last_ticket_date=max(closed_at,na.rm=TRUE), .groups="drop")
st_agg$sat_mean[is.na(st_agg$sat_mean)] <- mean(st_agg$sat_mean, na.rm=TRUE)
st_agg$days_since_last_ticket <- as.numeric(ref_date - st_agg$last_ticket_date)

df <- acc %>% select(account_id,churn,seats,plan_tier,referral_source,industry,account_tenure_days) %>%
  left_join(sub_agg, by="account_id") %>%
  left_join(fu_agg %>% select(-last_usage_date), by="account_id") %>%
  left_join(st_agg %>% select(-last_ticket_date), by="account_id")

fill_mean <- function(x){x[is.na(x)] <- mean(x,na.rm=TRUE); x}
df$tickets_n       <- ifelse(is.na(df$tickets_n),0,df$tickets_n)
df$resolution_hrs  <- fill_mean(df$resolution_hrs)
df$first_resp_mins <- fill_mean(df$first_resp_mins)
df$high_pri_share  <- ifelse(is.na(df$high_pri_share),0,df$high_pri_share)
df$escalation_rate <- ifelse(is.na(df$escalation_rate),0,df$escalation_rate)
df$days_since_last_ticket <- fill_mean(df$days_since_last_ticket)
df$days_since_last_usage  <- fill_mean(df$days_since_last_usage)

# Stage 4 features
stage1 <- c("resolution_hrs","first_resp_mins","high_pri_share","escalation_rate",
            "sat_mean","tickets_n","error_sum","usage_count_sum","usage_duration_sum",
            "unique_features","beta_share","sub_tenure_days_avg","sub_churn_rate",
            "account_tenure_days","mrr_mean","seats","upgrade_rate","downgrade_rate",
            "annual_share","auto_renew_share")
df$tickets_per_seat     <- df$tickets_n/(df$seats+1)
df$errors_per_usage     <- df$error_sum/(df$usage_count_sum+1)
df$usage_per_seat       <- df$usage_count_sum/(df$seats+1)
df$mrr_per_seat         <- df$mrr_mean/(df$seats+1)
df$support_hrs_per_seat <- (df$resolution_hrs*df$tickets_n)/(df$seats+1)
df$friction_index       <- df$escalation_rate*df$resolution_hrs
df$risk_composite       <- df$escalation_rate*df$high_pri_share
df$value_at_risk        <- df$mrr_mean*(df$escalation_rate+0.01)
df$engagement_rate      <- df$usage_count_sum/(df$account_tenure_days+1)
df$support_vs_usage     <- df$tickets_n/(df$usage_count_sum+1)
stage4 <- c(stage1,"tickets_per_seat","errors_per_usage","usage_per_seat","mrr_per_seat",
            "support_hrs_per_seat","friction_index","risk_composite","value_at_risk",
            "engagement_rate","support_vs_usage","days_since_last_usage","days_since_last_ticket")

cat_vars <- c("plan_tier","referral_source","industry")
build_X <- function(data, num_feats) {
  feats <- num_feats[num_feats %in% names(data)]
  X <- data[,feats,drop=FALSE]
  for (cv in cat_vars) {
    if (cv %in% names(data)) {
      dm <- model.matrix(as.formula(paste("~",cv,"-1")), data=data)
      X  <- cbind(X, dm)
    }
  }
  X[!is.finite(as.matrix(X))] <- 0
  as.matrix(X)
}
get_auc <- function(p,y) as.numeric(pROC::auc(pROC::roc(y,p,quiet=TRUE)))

set.seed(42)
churn_idx   <- which(df$churn==1); nochurn_idx <- which(df$churn==0)
train_idx   <- c(sample(churn_idx,floor(0.7*length(churn_idx))),
                 sample(nochurn_idx,floor(0.7*length(nochurn_idx))))
test_idx    <- setdiff(seq_len(nrow(df)), train_idx)
pos_weight  <- sum(df$churn[train_idx]==0)/sum(df$churn[train_idx]==1)

Xall4 <- build_X(df, stage4)
Xtr4  <- Xall4[train_idx,,drop=FALSE]; Xte4 <- Xall4[test_idx,,drop=FALSE]
ytr4  <- df$churn[train_idx];          yte4 <- df$churn[test_idx]
dtr_xgb <- xgb.DMatrix(data=Xtr4,label=ytr4)
dte_xgb <- xgb.DMatrix(data=Xte4,label=yte4)

# Best params from tuning
best_xgb  <- xgb_tune[which.max(xgb_tune$AUC),]
best_rf   <- rf_tune[which.max(rf_tune$AUC),]
best_lgbm <- lgbm_tune[which.max(lgbm_tune$AUC),]

# LR
d_tr <- as.data.frame(Xtr4); d_tr$y <- ytr4
d_te <- as.data.frame(Xte4)
safe <- make.names(names(d_tr))
names(d_tr) <- safe; names(d_te) <- safe[-length(safe)]
m_lr <- suppressWarnings(glm(y~., data=d_tr, family="binomial"))
auc_lr_f <- get_auc(predict(m_lr, newdata=d_te, type="response"), yte4)

# Elastic Net
cv_en <- cv.glmnet(Xtr4, ytr4, family="binomial", alpha=0.5, nfolds=5, type.measure="auc")
auc_enet_f <- get_auc(predict(cv_en,newx=Xte4,s="lambda.min",type="response")[,1], yte4)

# RF best
df_tr_rf <- as.data.frame(Xtr4); df_tr_rf$y <- as.factor(ytr4)
df_te_rf  <- as.data.frame(Xte4)
names(df_tr_rf) <- make.names(names(df_tr_rf))
names(df_te_rf)  <- make.names(names(df_te_rf))
m_rf <- ranger(y~., data=df_tr_rf, num.trees=500, probability=TRUE,
               mtry=best_rf$mtry, min.node.size=best_rf$min_node,
               class.weights=c("0"=1,"1"=pos_weight), seed=42)
auc_rf_f <- get_auc(predict(m_rf,data=df_te_rf)$predictions[,"1"], yte4)

# XGBoost — use nrounds=100 minimum to avoid 1-round model
nrounds_xgb <- max(best_xgb$best_rounds, 100)
cat(sprintf("XGBoost final: depth=%d eta=%.2f sub=%.1f nrounds=%d\n",
            best_xgb$depth, best_xgb$eta, best_xgb$subsample, nrounds_xgb))
m_xgb <- xgb.train(
  params=list(objective="binary:logistic", eval_metric="auc",
              max_depth=best_xgb$depth, eta=best_xgb$eta,
              subsample=best_xgb$subsample, colsample_bytree=0.8,
              scale_pos_weight=pos_weight),
  data=dtr_xgb, nrounds=nrounds_xgb, verbose=0)
auc_xgb_f <- get_auc(predict(m_xgb, dte_xgb), yte4)

# LightGBM best
dtr_lgb <- lgb.Dataset(Xtr4, label=ytr4)
m_lgbm <- lgb.train(
  params=list(objective="binary", metric="auc",
              num_leaves=best_lgbm$num_leaves, learning_rate=best_lgbm$lr,
              scale_pos_weight=pos_weight, verbose=-1, min_data_in_leaf=5),
  data=dtr_lgb, nrounds=300, verbose=-1)
auc_lgbm_f <- get_auc(predict(m_lgbm, Xte4), yte4)

# Compile
final_cmp <- data.frame(
  Model    = c("LR (A2 baseline)","LR (S4)",
               "Elastic Net (S4)","RF (A2 baseline)","RF (S4, tuned)",
               "XGBoost (A2 baseline)","XGBoost (S4, tuned)",
               "LightGBM (S4, tuned)"),
  AUC      = c(0.5730, auc_lr_f, auc_enet_f,
               0.5915, auc_rf_f,
               0.5664, auc_xgb_f, auc_lgbm_f),
  Category = c("A2 Baseline","Stage 4","Stage 4",
               "A2 Baseline","Stage 4 Tuned",
               "A2 Baseline","Stage 4 Tuned","Stage 4 Tuned"),
  stringsAsFactors=FALSE)

final_cmp <- final_cmp[order(-final_cmp$AUC),]
cat("\n=== FINAL MODEL COMPARISON ===\n")
print(final_cmp, row.names=FALSE, digits=4)
write.csv(final_cmp, file.path(out,"a3_final_model_comparison.csv"), row.names=FALSE)

# Rebuild plot
fc <- final_cmp
fc$Model    <- factor(fc$Model, levels=fc$Model[order(fc$AUC)])
fc$Category <- factor(fc$Category, levels=c("A2 Baseline","Stage 4","Stage 4 Tuned"))

p <- ggplot(fc, aes(x=Model, y=AUC, fill=Category)) +
  geom_col(width=0.65) +
  geom_text(aes(label=sprintf("%.4f",AUC)), hjust=-0.1, size=4) +
  geom_hline(yintercept=0.5, linetype="dashed", color="gray60", linewidth=0.8) +
  scale_fill_manual(values=c("A2 Baseline"="#90A4AE","Stage 4"="#64B5F6","Stage 4 Tuned"="#1565C0")) +
  coord_flip() + scale_y_continuous(limits=c(0,0.85)) +
  labs(title="Full Model Comparison: A2 Baseline vs Stage 4 + Tuned",
       subtitle="All Stage 4 models use enriched features (ratios, interactions, recency)",
       x=NULL, y="AUC (Test Set)", fill="Configuration",
       caption="Dashed line = random chance (0.50)") +
  theme_minimal(base_size=13) + theme(plot.title=element_text(face="bold"))

ggsave(file.path(out,"a3_model_comparison_tuned.png"), p, width=10, height=6, dpi=150)
cat("Plot saved.\n")
