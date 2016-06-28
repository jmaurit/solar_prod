#descriptives.py

import pandas as pd
import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm


pd.options.display.max_rows = 999
pd.options.display.max_columns = 50


#larger sample but fewer covariates
prod_long=pd.read_csv("research/solar_prod/prod_long.csv")
prod_long["prod_end_date"]=pd.to_datetime(prod_long["prod_end_date"])



#data file with many covariates, but limited sample
prod_wide=pd.read_csv("research/solar_prod/prod_wide.csv")
prod_wide["prod_end_date"]=pd.to_datetime(prod_wide["prod_end_date"])
prod_wide["complete_date"]=pd.to_datetime(prod_wide["complete_date"])

wide_cross = prod_wide.groupby("app_num").apply(lambda x: x.iloc[0])
wide_cross.groupby("program").size()

#just the individual cells




#create max-production variable
prod_long.reset_index(inplace=True)
prod_long["prod_max"]=prod_long.groupby("app_num")["prod_kwh"].transform(max)
prod_long["prod_index"] = prod_long["prod_kwh"]/prod_long["prod_max"]

#create months of operation variable
count_index = lambda v: [i+1 for i, x in enumerate(v)]
prod_long["months_operation"] = prod_long["index"].groupby(prod_long["app_num"]).transform(count_index)

prod_long["month"] = prod_long["prod_end_date"].dt.month
month_dummies = pd.get_dummies(prod_long["month"])
month_dummies.columns = ["jan","feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct" ,"nov", "dec"]
prod_long["year"]=prod_long["prod_end_date"].dt.year
year_dummies = pd.get_dummies(prod_long["year"])
year_dummies.columns = ["y2007", "y2008", "y2009","y2010", "y2011", "y2012", "y2013", "y2014", "y2015"]

solar_data=pd.concat([prod_long, month_dummies], axis=1)
solar_data=pd.concat([solar_data, year_dummies], axis=1)

solar_data["installation_year"] = solar_data.groupby("app_num")["year"].transform(np.min)

solar_data["installation_year"] = solar_data["installation_year"]-2007

#check for bad data
solar_data[["app_num", "prod_kwh"]].head(2000)

#scatter plots

fig, ax = plt.subplots()
ax.scatter(solar_data["months_operation"], solar_data["prod_index"], alpha=.01)
plt.show()


#model with fixed effects (pooled model)
fe_formula = """prod_index ~ months_operation*installation_year +
 cust_county + feb + mar + apr + may + jun + jul + aug + sep + 
 oct + nov + dec + y2008 + y2009 + y2010 + y2011 + y2012 +
  y2013 + y2014 + y2015"""
fe_model = smf.glm(formula=fe_formula, data=solar_data, 
	family=sm.families.Gaussian(sm.families.links.log)).fit()
fe_model.summary()

del solar_data["index"]

mixed_model = smf.mixedlm("prod_index ~ months_operation", solar_data, 
 groups=solar_data["host_cust_sect"], missing="drop")
mixed_model_fit = mixed_model.fit()
predicted = fitted.predict()

fig, ax = plt.subplots()
ax.plot(single.months_operation.iloc[1:], predicted)
ax.plot(single.months_operation, single.prod_index)
plt.show()

#group several
prod_long=prod_long.set_index(["app_num", "prod_end_date"])
prod_long=prod_long.sort()
multi=prod_long.loc['PGE-CSI-00001':'PGE-CSI-00010'].copy()

multi.reset_index(inplace=True)
multi=multi[multi.app_num!='PGE-CSI-00007']

	
fig, ax = plt.subplots()
for system, i in multi.groupby("app_num"):
	ax.plot(i["prod_end_date"], i["prod_index"])
ax.set_ylabel("Production Index")

plt.savefig("research/solar_prod/figures/prod10.png", dpi=150)
plt.show()

#create month


multi["months"] = multi["index"].groupby(multi["app_num"]).transform(count_index)

sns.lmplot(x="months", y="prod_index", col="app_num", 
	data=multi, col_wrap=3)


#For full dataset

sns.lmplot(x="months_operation", y="prod_index", 
	data=prod_long, hue="own_sect", scatter=False)
plt.show()

sns.lmplot(x="months_operation", y="prod_index", 
	data=prod_long, hue="program", scatter=False)
plt.show()

sns.lmplot(x="months_operation", y="prod_index", 
	data=prod_long, row="program", col = "own_sect", scatter=False)
plt.show()

#some general descriptives
prod_long.groupby(["program"]).describe()

#get cross-section of data
prod_long.reset_index(inplace=True)
cross_section=prod_long.groupby("app_num").apply(lambda x: x.iloc[0])
del cross_section["app_num"]
cross_section.reset_index(inplace=True)
cross_section=cross_section.rename(columns={'prod_end_date': 'first_prod', 
	'prod_kwh': 'avg_prod_kwh'})
cross_section["avg_prod_kwh"]=prod_long.groupby("app_num")["prod_kwh"].mean()

cross_section.groupby("program").size()

fig, ax = plt.subplots()
ax.hist(cross_section["tot_cost"]/1000, bins=50)
ax.set_xlabel("Cost of System, 1000 USD")
plt.show()

#average production
fig, ax = plt.subplots()
ax.hist(cross_section["avg_prod_kwh"]/1000, bins=50)
ax.set_xlabel("Average weekly production,1000 kwh")
ax.set_xlim([0,250])
plt.show()

#cost_per_max_kwh
cross_section["cost_per_max_kwh"]=cross_section["tot_cost"]/cross_section["prod_max"]
solar_data = solar_data.merge(cross_section[["app_num", "cost_per_max_kwh"]], how="left")
solar_data.to_csv("research/solar_prod/solar_data.csv")


cross_section.sort("first_prod", inplace=True)
cross_section.to_csv("research/solar_prod/solar_cross_section.csv")

sectors= cross_section["own_sect"].unique()
colors=["blue", "red", "green", "yellow"]
sectorsZ=zip(sectors.tolist(), colors)

fig, ax = plt.subplots()
fig.set_size_inches(9,7)
for sect in sectorsZ:
	own_sect, color = sect
	ax.plot_date(x=cross_section.loc[cross_section["own_sect"]==own_sect,"first_prod"], 
		y=cross_section.loc[cross_section["own_sect"]==own_sect,"cost_per_max_kwh"],
		alpha=.2, color=color, label=own_sect)
	col_patch = mpatches.Patch(color=color, label=own_sect)
ax.set_ylabel("Cost per max kwh")
ax.set_ylim([0,100])
plt.legend()
plt.savefig("research/solar_prod/figures/cost_per_max_kwh.png", dpi=150)
plt.show()

#using seaborn

#are newer ones better over time?

#are chinese worse than others?




