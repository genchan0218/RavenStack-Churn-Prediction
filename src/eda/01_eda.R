# =========================================================
# A.1 Interim Report Analysis Script
# Project: RavenStack SaaS Churn & Retention Signals
# =========================================================

# ---- Libraries ----
libs <- c("data.table", "dplyr", "lubridate", "ggplot2", "stringr", "broom", "tidyr")
invisible(lapply(libs, require, character.only = TRUE))

# ---- File paths ----
p_accounts <- "C:/Users/Genki/Documents/Homework/AI_Capstone/ravenstack_accounts.csv"
p_subs     <- "C:/Users/Genki/Documents/Homework/AI_Capstone/ravenstack_subscriptions.csv"
p_usage    <- "C:/Users/Genki/Documents/Homework/AI_Capstone/ravenstack_feature_usage.csv"
p_tickets  <- "C:/Users/Genki/Documents/Homework/AI_Capstone/ravenstack_support_tickets.csv"
p_churn    <- "C:/Users/Genki/Documents/Homework/AI_Capstone/ravenstack_churn_events.csv"

# ---- Load ----
accounts <- fread(p_accounts)
subs     <- fread(p_subs)
usage    <- fread(p_usage)
tickets  <- fread(p_tickets)
churn    <- fread(p_churn)

# ---- Basic shape check ----
cat("Rows x Cols\n")
cat("accounts:", nrow(accounts), "x", ncol(accounts), "\n")
cat("subs    :", nrow(subs), "x", ncol(subs), "\n")
cat("usage   :", nrow(usage), "x", ncol(usage), "\n")
cat("tickets :", nrow(tickets), "x", ncol(tickets), "\n")
cat("churn   :", nrow(churn), "x", ncol(churn), "\n")

# ---- Parse dates ----
accounts[, signup_date := as.Date(signup_date)]
subs[, `:=`(start_date = as.Date(start_date),
            end_date   = as.Date(end_date))]
usage[, usage_date := as.Date(usage_date)]
tickets[, `:=`(
  submitted_at = ymd_hms(submitted_at, quiet = TRUE),
  closed_at    = ymd_hms(closed_at, quiet = TRUE)
)]
churn[, churn_date := as.Date(churn_date)]

# ---- Null summary helper ----
null_summary <- function(dt, nm) {
  data.frame(
    table = nm,
    column = names(dt),
    null_n = sapply(dt, function(x) sum(is.na(x))),
    null_pct = round(100 * sapply(dt, function(x) mean(is.na(x))), 2)
  )
}
nulls <- bind_rows(
  null_summary(accounts, "accounts"),
  null_summary(subs, "subscriptions"),
  null_summary(usage, "feature_usage"),
  null_summary(tickets, "support_tickets"),
  null_summary(churn, "churn_events")
)
print(nulls %>% arrange(desc(null_pct)) %>% head(20))

# ---- Build account-level modeling table ----
# 1) feature usage aggregated at subscription level
usage_sub <- usage %>%
  group_by(subscription_id) %>%
  summarise(
    usage_events       = n(),
    usage_count_sum    = sum(usage_count, na.rm = TRUE),
    usage_duration_sum = sum(usage_duration_secs, na.rm = TRUE),
    error_sum          = sum(error_count, na.rm = TRUE),
    beta_share         = mean(is_beta_feature, na.rm = TRUE),
    .groups = "drop"
  )

# 2) subscriptions + usage -> account level
sub_w_usage <- subs %>%
  left_join(usage_sub, by = "subscription_id") %>%
  mutate(
    across(c(usage_events, usage_count_sum, usage_duration_sum, error_sum, beta_share),
           ~replace_na(., 0))
  )

acc_sub <- sub_w_usage %>%
  group_by(account_id) %>%
  summarise(
    subs_n           = n(),
    mrr_mean         = mean(mrr_amount, na.rm = TRUE),
    arr_mean         = mean(arr_amount, na.rm = TRUE),
    annual_share     = mean(billing_frequency == "annual", na.rm = TRUE),
    auto_renew_share = mean(auto_renew_flag, na.rm = TRUE),
    upgrade_rate     = mean(upgrade_flag, na.rm = TRUE),
    downgrade_rate   = mean(downgrade_flag, na.rm = TRUE),
    sub_churn_rate   = mean(churn_flag, na.rm = TRUE),
    usage_events     = sum(usage_events, na.rm = TRUE),
    usage_count_sum  = sum(usage_count_sum, na.rm = TRUE),
    usage_duration_sum = sum(usage_duration_sum, na.rm = TRUE),
    error_sum        = sum(error_sum, na.rm = TRUE),
    beta_share       = mean(beta_share, na.rm = TRUE),
    .groups = "drop"
  )

# 3) tickets aggregated at account level
acc_tickets <- tickets %>%
  group_by(account_id) %>%
  summarise(
    tickets_n        = n(),
    resolution_hrs   = mean(resolution_time_hours, na.rm = TRUE),
    first_resp_mins  = mean(first_response_time_minutes, na.rm = TRUE),
    escalation_rate  = mean(escalation_flag, na.rm = TRUE),
    sat_mean         = mean(satisfaction_score, na.rm = TRUE),
    .groups = "drop"
  )

# 4) churn events aggregated at account level
acc_churn_evt <- churn %>%
  group_by(account_id) %>%
  summarise(
    churn_events_n = n(),
    refund_sum     = sum(refund_amount_usd, na.rm = TRUE),
    .groups = "drop"
  )

# 5) merge all
model_df <- accounts %>%
  left_join(acc_sub, by = "account_id") %>%
  left_join(acc_tickets, by = "account_id") %>%
  left_join(acc_churn_evt, by = "account_id") %>%
  mutate(
    across(c(tickets_n, resolution_hrs, first_resp_mins, escalation_rate, sat_mean,
             churn_events_n, refund_sum), ~replace_na(., 0)),
    target_churn = as.integer(churn_flag)
  )

cat("\nAccount-level modeling rows:", nrow(model_df), "\n")
cat("Churn prevalence:", round(mean(model_df$target_churn), 4), "\n")

# ---- Quick descriptive comparison ----
desc <- model_df %>%
  group_by(target_churn) %>%
  summarise(
    n = n(),
    seats_mean = mean(seats, na.rm = TRUE),
    mrr_mean = mean(mrr_mean, na.rm = TRUE),
    usage_mean = mean(usage_count_sum, na.rm = TRUE),
    errors_mean = mean(error_sum, na.rm = TRUE),
    tickets_mean = mean(tickets_n, na.rm = TRUE),
    escalation_rate = mean(escalation_rate, na.rm = TRUE),
    sat_mean = mean(sat_mean, na.rm = TRUE),
    .groups = "drop"
  )
print(desc)

# ---- Initial baseline model (interpretable logistic) ----
# Keep simple for A.1 (can expand in A.2)
model_df2 <- model_df %>%
  mutate(
    industry = as.factor(industry),
    referral_source = as.factor(referral_source),
    plan_tier = as.factor(plan_tier),
    country = as.factor(country)
  )

fit <- glm(
  target_churn ~ seats + mrr_mean + usage_count_sum + error_sum + tickets_n +
    escalation_rate + sat_mean + downgrade_rate + upgrade_rate +
    auto_renew_share + industry + referral_source + plan_tier,
  data = model_df2,
  family = binomial(link = "logit")
)

summary(fit)

# odds ratios table
or_tbl <- broom::tidy(fit, conf.int = TRUE, exponentiate = TRUE) %>%
  arrange(p.value)
print(head(or_tbl, 20))

# ---- Save analysis artifacts ----
write.csv(nulls, "C:/Users/Genki/Documents/Homework/AI_Capstone/reports/a1_null_summary.csv", row.names = FALSE)
write.csv(desc, "C:/Users/Genki/Documents/Homework/AI_Capstone/reports/a1_churn_descriptives.csv", row.names = FALSE)
write.csv(or_tbl, "C:/Users/Genki/Documents/Homework/AI_Capstone/reports/a1_logit_odds_ratios.csv", row.names = FALSE)

cat("\nSaved:\n",
    "C:/Users/Genki/Documents/Homework/AI_Capstone/a1_null_summary.csv\n",
    "C:/Users/Genki/Documents/Homework/AI_Capstone/a1_churn_descriptives.csv\n",
    "C:/Users/Genki/Documents/Homework/AI_Capstone/a1_logit_odds_ratios.csv\n")
