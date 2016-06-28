#analysis.r

rm(list = ls())

library(ggplot2)
library(reshape2)
library(dplyr)
library(plyr) # upgrade to dplyr
library(arm) #special functions for Gelman & Hill
library(lme4)
library(mgcv)
library(texreg)
library(lubridate) #for extracting date
# library(rstan)
library(sandwich) #robust standard errors
library(grid)

solar_data <- read.csv("/Users/johannesmauritzen/research/solar_prod/solar_data.csv", stringsAsFactor = FALSE)

#lmer model
mlm_production<-lmer(log(prod_index + .0001) ~ months_operation + factor(month) + factor(year) + installation_year + own_sect*months_operation + 
	(1+months_operation|app_num), data=solar_data)
summary(mlm_production)

mlm_production2<-lmer(log(prod_index + .0001) ~ months_operation + factor(month) + factor(year) + installation_year + onw_sect + cost_per_max_kwh*months_operation + 
	(1+months_operation|app_num), data=solar_data)
summary(mlm_production2)


reg_prod_coef <- fixef(mlm_production2)
reg_prod_se <- se.fixef(mlm_production2)
reg_prod <- data.frame(reg_prod_coef, reg_prod_se)


#simulations from posterior:
mlm_prod_sim <- sim(mlm_production2)
meta_sim1 <- mlm_prod_sim@fixef
meta_sim1 <- as.data.frame(meta_sim1)

a_hat<-coef(mlm_production2)$app_num[,1]
b_hat<-coef(mlm_production2)$app_num[,2]
a_se<-se.ranef(mlm_production2)$app_num[,1]
b_se<-se.ranef(mlm_production2)$app_num[,2]

app_num <- rownames(coef(mlm_production)$app_num)

cross_section<-read.csv("/Users/johannesmauritzen/research/solar_prod/solar_cross_section.csv", stringsAsFactor = FALSE)


ab_s <-data.frame(a_hat = a_hat, b_hat=b_hat, 
	a_se=a_se, b_se=b_se, app_num = app_num)

reg_data <- merge(ab_s, cross_section, by="app_num")
reg_data <- reg_data[order(reg_data$b_hat),]

production_slope<-ggplot(reg_data) +
	geom_pointrange(aes(x=own_sect, y = b_hat, 
		ymin=b_hat - 2*b_se, 
		ymax = b_hat + 2*b_se), position="jitter", alpha=.1)


solar_data[]

production_slope


