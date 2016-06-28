#analysis_wide.r

rm(list = ls())

library(ggplot2)
library(reshape2)
library(dplyr)
library(arm) #special functions for Gelman & Hill
library(lme4)
library(mgcv)
library(texreg)
library(lubridate) #for extracting date
# library(rstan)
library(sandwich) #robust standard errors
library(grid)
#library(plm) #for panel data

prod_wide <- read.csv("/Users/johannesmauritzen/research/solar_files/prod_wide_clean.csv", stringsAsFactor = FALSE)

#linear model

prod_wide <- prod_wide[!is.na(prod_wide$prod_index),]
prod_wide <- prod_wide[!is.na(prod_wide$months_operation),]



prod_wide$log_prod_kwh_st <- (log(prod_wide$prod_kwh) - mean(log(prod_wide$prod_kwh)))/sd(log(prod_wide$prod_kwh))
prod_wide$months_operation_st <- prod_wide$months_operation - mean(prod_wide$months_operation)

lm_prod <- lm(prod_index ~ months_operation*lease + 
	factor(month)*factor(year), data=prod_wide)
summary(lm_prod)

lm_prod2 <- lm(log_prod_kwh_st ~ I(months_operation_st/12)*factor(third_party_owned) + sys_size_dc +
	factor(cust_county) + factor(month) + factor(tracking) + I(months_operation_st/12)*factor(sector), data=prod_wide)
summary(lm_prod2)

texreg(lm_prod2, robust=TRUE, digits=3)

plm_prod2 <- plm(I(log(prod_kwh)) ~ I(months_operation/12)*factor(third_party_owned) + 
	sys_size_dc + factor(cust_county) + factor(month)  + factor(tracking) + factor(sector), data=prod_wide, model = "pooling")
summary(plm_prod2)



mlm_production<-lmer(log(prod_index + .0001) ~ months_operation*lease + factor(month) + 
	(1+months_operation|app_num), data=prod_wide)
summary(mlm_production)

reg_new_coef<-fixef(mlm_production)
reg_new_se<-se.fixef(mlm_production)
reg_new<-data.frame(reg_new_coef, reg_new_se)

mlm_production_sim<-sim(mlm_production, 1000)
meta_sim1<-mlm_production_sim@fixef
meta_sim1<-as.data.frame(meta_sim1)

a_hat<-coef(mlm_production)$app_num[,1]
b_hat<-coef(mlm_production)$app_num[,2]
a_se<-se.ranef(mlm_production)$app_num[,1]
b_se<-se.ranef(mlm_production)$app_num[,2]

app_num<-rownames(coef(mlm_production)$app_num)

app_data <- prod_wide[, c("app_num", "lease")]
app_data <- app_data[!duplicated(app_data$app_num),]

app_num <- data.frame(app_num=app_num)
app_num <- merge(app_num, app_data, by.x=TRUE, all.x=TRUE)

ab_s<-data.frame(a_hat=a_hat, b_hat=b_hat, 
	a_se=a_se, b_se=b_se, app_num=app_num$app_num, lease = app_num$lease)

meta_sim1["slope_lease"]<-meta_sim1["months_operation"] + meta_sim1["months_operation:lease"]

ggplot(ab_s) +
	geom_pointrange(aes(x=lease, 
	y=b_hat, 
	ymin=b_hat-2*b_se,
	ymax=b_hat+2*b_se),
	position="jitter") +
	geom_hline(aes(yintercept=months_operation), data=meta_sim1, color="grey", alpha=.1) +
	geom_hline(yintercept=reg_new["months_operation",1]) +
	geom_hline(aes(yintercept=slope_lease),
		data=meta_sim1, color="red", alpha=.05) +
	geom_hline(yintercept=reg_new["months_operation",1] + reg_new["months_operation:lease",1])
	#labs(x="Change in percent of installations \n that are leased and use Chinese panels", y="Contractor level estimated slope of price fall")


meta_sim1["diff_slope"] <- meta_sim1["slope_lease"] - meta_sim1["months_operation"]

ggplot(meta_sim1) +
	geom_histogram(aes(x=diff_slope))

#prediction: show prediction between leased and non-leased.




