#lmel_modelling.r
#Modelling data using r package lmel

install.packages("tidyverse")
install.packages("gamm4")
install.packages("stargazer")

load_libraries = function(){
  library(lme4)
  library(tidyverse)
  library(gamm4)
  library(mgcv)
  library(arm)
  library(stargazer)

}

#initial functions
load_data = function(){
  solar_data = read.csv("data/prod_csi.csv")
  return(solar_data)
}

show_production = function(solar_data){
  prod_graph =
  ggplot(solar_data, aes(x=months_operation, y=prod_scaled, group = app_num)) +
  geom_line(alpha=.01)
  return(prod_graph)
}

format_solar = function(solar_data){
  solar_data["month"] = as.factor(solar_data$month)
  solar_data = solar_data[solar_data$module_manufacturer!="Chaori", ]
  #at least three years of data
  solar_data = solar_data[solar_data$total_months_operation>=36,]
  return(solar_data)
}

#helper functions
extract_slopes = function(lme_model){
  #test
  #lme_model = solar_sys
  #
  slope_fe = fixef(lme_model)[13]

  slope_re=data.frame(ranef(lme_model))

  levels(slope_re$term) = c("intercept", "slope")
  #colnames(slope_re) = c("intercept", "slope")
  slopes = slope_fe + slope_re$condval[slope_re$term=="slope"]

  se_re = se.ranef(lme_model)

  if(length(se_re)>1){
    i=1
    slope_ses <- list(NA, length(se_re))

    for (name in names(se_re)){
      se_re_df=data.frame(se_re[name])
      colnames(se_re_df) = c("intercept_se", "slope_se")
      slope_ses[[i]] = se_re_df[2]
      i=i+1
    }
    slope_ses_df = do.call("rbind", slope_ses)
  } else{
    se_re_df = data.frame(se_re[1])
    colnames(se_re_df) = c("intercept_se", "slope_se")
    slope_ses_df = se_re_df[2]
  }

  slopes_df = data.frame(slopes)
  slopes_df["se"] = slope_ses_df["slope_se"]

  slopes_df["pos_ci"] = slopes_df["slopes"] + 2*slopes_df["se"]
  slopes_df["neg_ci"] = slopes_df["slopes"] - 2*slopes_df["se"]
  slopes_df["names"] = factor(row.names(slope_ses_df))

  slopes_df["grpvar"] = slope_re[slope_re$term=="slope","grpvar"]
  return(slopes_df)
}

order_slopes = function(slopes_df, ordervar = slopes_df$slope){
  slopes_df["names"] = factor(slopes_df$names, levels=slopes_df$names[order(ordervar)])
  return(slopes_df)
}

#model plotting:
slope_plot = function(slopes_df, no_xlabs=FALSE, order_slopes=TRUE, alpha=1){
  if(order_slopes){
    slopes_df = order_slopes(slopes_df)
  }
  slopes_figure = ggplot(slopes_df, aes(x=names, y=slopes)) +
    geom_errorbar(aes(ymax=pos_ci, ymin=neg_ci), alpha=alpha)
  if(no_xlabs){
    return(slopes_figure + theme(axis.ticks.x=element_blank(), axis.text.x=element_blank()))
  }else{
    return(slopes_figure+theme(axis.text = element_text(angle = 90)))
  }
}





#simple model with only lease or buy
simple_model = function(solar_data){
  solar_lme = lmer(prod_scaled ~ month + months_operation + (months_operation | third_party_owned), solar_data)
  return(solar_lme)
}

#Simple model with manufacturers
lme2_model = function(solar_data){
  solar_lme2 = lmer(prod_scaled ~ month + months_operation_scaled + (months_operation_scaled |module_manufacturer), solar_data)
  return(solar_lme2)
}

#Gam - use gam to capture seasonal effects.
gam_model = function(solar_data){

}

#nested model with modeling of individual systems
sys_model = function(solar_data){
  solar_sys = lmer(prod_scaled ~ month + months_operation_scaled + (months_operation_scaled |app_num), solar_data)
  return(solar_sys)
}

#also with manufacturer
nested_model = function(solar_data){
  solar_lme3 = lmer(prod_scaled ~ month + months_operation_scaled + (months_operation_scaled |module_manufacturer) + (months_operation_scaled|module_manufacturer:app_num), solar_data)
  return(solar_lme3)
}

lease_model = function(solar_data){
  solar_lme4 = lmer(prod_scaled ~ month + months_operation_scaled + (months_operation_scaled |third_party_owned) + (months_operation_scaled|third_party_owned:app_num), solar_data)
  return(solar_lme4)
}

lease_model2 = function(solar_data){
  solar_lme4 = lmer(prod_scaled ~ month + months_operation_scaled + cost_per_kw (months_operation_scaled + cost_per_kw |third_party_owned) + (months_operation_scaled|third_party_owned:app_num), solar_data)
  return(solar_lme4)
}

main = function(){
  load_libraries()
  solar_data = load_data()
  solar_data = format_solar(solar_data)
  prod_fig = show_production(solar_data)

  lm_model = lm(prod_scaled~months_operation, data=solar_data)
  lm_coef = coef(lm_model)

  prod_fig + geom_abline(intercept=lm_coef[1], slope=lm_coef[2], color="red") + xlim(0,60) + ylim(-1.5,1.5) + labs(x="months",y="Production, de-scaled")

  ggsave("figures/production.png", plot=prod_fig, dpi=300)

  #model with manufacturers
  solar_lme2 = lme2_model(solar_data)
  slopes_df_man = extract_slopes(solar_lme2)
  man_fig = slope_plot(slopes_df_man, no_xlabs=TRUE)
  man_fig_final = man_fig + geom_hline(yintercept=0, color="red") + labs(x="Module Manufacturers", y="Slope over time, descaled")
  man_fig_final
  ggsave("figures/manufacturer_slope.png", plot=man_fig_final, dpi=300)

  #model with systems
  solar_sys = sys_model(solar_data)
  slopes_df_sys = extract_slopes(solar_sys)
  sys_slope_fig = slope_plot(slopes_df_sys, no_xlabs=TRUE)
  sys_slope_fig_final = sys_slope_fig + geom_hline(yintercept=0, color="red") + labs(x="Solar panel systems", y="Slope over time, descaled")
  sys_slope_fig_final
  ggsave("figures/sys_slope_fig.png", plot=sys_slope_fig_final, dpi=300)

  #nested model with manufacturer and system
  solar_nested = nested_model(solar_data)
  slopes_df_nest = extract_slopes(solar_nested)
  nested_manuf_fig = slope_plot(slopes_df_nest[slopes_df_nest$grpvar=="module_manufacturer", ], no_xlabs=TRUE)
  nested_manuf_fig_final = nested_manuf_fig + geom_hline(yintercept=0, color="red") + labs(x="Module Manufacturers", y="Slope over time, descaled")
  nested_manuf_fig_final

  ggsave("figures/nested_manuf_fig_final.png", plot=nested_manuf_fig_final, dpi=300)


  nested_sys_fig = slope_plot(slopes_df_nest[slopes_df_nest$grpvar=="module_manufacturer:app_num", ], no_xlabs=TRUE)

  nested_sys_fig_final = nested_sys_fig + geom_hline(yintercept=0, color="red") + labs(x="Solar panel systems", y="Slope over time, descaled")

  nested_sys_fig_final

  ggsave("figures/nested_sys_fig.png", plot=nested_sys_fig_final, dpi=300)

#Now model also with lease no lease:

  solar_lease = lease_model(solar_data)

  slopes_df_lease = extract_slopes(solar_lease)

  slopes_df_lease["names"]= as.character(slopes_df_lease$names)

  slopes_df_lease = slopes_df_lease %>% separate(names, c("third_party_owned", "names"), sep=":")

  #manually change ordering:
  slopes_df_lease["names"] = factor(slopes_df_lease$names, levels=slopes_df_lease$names[order(slopes_df_lease$third_party_owned, slopes_df_lease$slopes)])

  leased_re_slopes = slopes_df_lease[slopes_df_lease$grpvar=="third_party_owned",]
  leased_re_slopes
  slopes_df_lease =  slopes_df_lease[slopes_df_lease$grpvar=="third_party_owned:app_num",]

  re_slopes_list = list("no" = leased_re_slopes$slopes[leased_re_slopes$third_party_owned=="no"],
  "yes" = leased_re_slopes$slopes[leased_re_slopes$third_party_owned=="yes"]
)

  re_se_list = list("no" = leased_re_slopes$se[leased_re_slopes$third_party_owned=="no"],
  "yes" = leased_re_slopes$se[leased_re_slopes$third_party_owned=="yes"]
  )

  slopes_df_lease["leased_re"] =  unlist(re_slopes_list[slopes_df_lease$third_party_owned])

  slopes_df_lease["leased_re_se"] =  unlist(re_se_list[slopes_df_lease$third_party_owned])
  slopes_df_lease["third_party_owned"] = factor(slopes_df_lease$third_party_owned)
  slopes_df_lease["re_ci_plus"] = slopes_df_lease["leased_re"] + 2* slopes_df_lease["leased_re_se"]
  slopes_df_lease["re_ci_minus"] = slopes_df_lease["leased_re"] - 2* slopes_df_lease["leased_re_se"]

  str(slopes_df_lease)


  lease_sys_fig = slope_plot(slopes_df_lease, order_slopes=FALSE,  no_xlabs=TRUE, alpha=.1)

  lease_sys_fig_final = lease_sys_fig +
  geom_line(aes(x=names, y=leased_re, linetype=third_party_owned, group=third_party_owned)) +
  geom_ribbon(aes(ymin=re_ci_minus, ymax=re_ci_plus, group=third_party_owned) ,alpha=.5) +
  geom_hline(yintercept=0, color="red") + labs(x="Solar panel systems", y="Slope over time, descaled", linetype="Leased")

  ggsave("figures/lease_sys_fig.png", plot=lease_sys_fig_final, dpi=300)

  

  #With varying prices as well.


  #table results of nested and non-nested
  stargazer(solar_sys, solar_nested)
  #Compare AIC scores and anova

  AIC(solar_sys, solar_nested, solar_lease)

  anova_results = anova(solar_sys, solar_nested, solar_lease)
  stargazer(anova_results)

  anova(solar_nested)

}
