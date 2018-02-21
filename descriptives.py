#descriptives.py
#%%
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


# %%
def set_graphics_params():
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['axes.facecolor'] ="white"
    plt.rcParams['grid.color'] ="grey"
    plt.rcParams['grid.linestyle'] = "dotted"
    plt.rcParams["axes.labelsize"]= 14
    #plt.rcParams['figure.savefig.dpi'] = 100
    plt.rcParams['savefig.edgecolor'] = "#f2f2f2"
    plt.rcParams['savefig.facecolor'] ="white"
    plt.rcParams["figure.figsize"] = [15,8]
    plt.rcParams['savefig.bbox'] = "tight"
    plt.rcParams['font.size'] = 14
    greens = ['#66c2a4','#41ae76','#238b45','#006d2c','#00441b']
    multi =['#66c2a4','#1f78b4','#a6cee3','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f']

#%%

set_graphics_params()
solar_data = pd.read_csv("data/prod_csi.csv")

comm_table = [["Model",	"DF",	"AIC",	"BIC",	"logLik",	"deviance",	"Chisq",	"Chi df",	"Pr(>Chisq)"],
["Null",	"17",	"8805",	"8976",	"-4385",	"8771",	"--",	"--",	"--"],
["Grouped Manuf.", 	"20",	"8430",	"8631",	"-4195",	"8390",	"381",	"3",	"2.6e-82"]]

comm_table_df = pd.DataFrame(comm_table)
comm_table_df.to_latex(buf="table3.tex")

#data file with many covariates, but limited sample
prod_wide=pd.read_csv("prod_wide.csv")
prod_wide.head()

prod_wide["prod_end_date"]=pd.to_datetime(prod_wide["prod_end_date"])
#prod_wide["complete_date"]=pd.to_datetime(prod_wide["complete_date"])


wide_cross = prod_wide.groupby("app_num").apply(lambda x: x.iloc[0])
wide_cross.groupby("program").size()


#%%

grouped_solar = solar_data.groupby("app_num")

fig,ax=plt.subplots()
for plant_data in grouped_solar:
    ax.plot(plant_data[1]["months_operation"], plant_data[1]["prod_index"], alpha=.1, color="blue")
plt.show()

#%%




#%%
#model with fixed effects (pooled model)
fe_formula = """prod_index ~ months_operation*installation_year +
 cust_county + feb + mar + apr + may + jun + jul + aug + sep +
 oct + nov + dec"""
fe_model = smf.glm(formula=fe_formula, data=solar_data,
	family=sm.families.Gaussian(sm.families.links.log)).fit()
fe_model.summary()


del solar_data["index"]


#Mixed Modelling:
#%%
mixed_model = smf.mixedlm("prod_index ~ months_operation + feb + mar + apr + may + jun + jul + aug + sep + oct + nov + dec", solar_data,
 groups=solar_data["host_cust_sect"], missing="drop")
mixed_model_fit = mixed_model.fit()
predicted = mixed_model_fit.predict()

#%%
result_DF = solar_data.copy()
result_DF["predicted1"] = predicted
result_DF.head()
#%%

fig, ax = plt.subplots()
ax.plot(single.months_operation.iloc[1:], predicted)
ax.plot(single.months_operation, single.prod_index)
plt.show()

#group several
prod_wide=prod_wide.set_index(["app_num", "prod_end_date"])
prod_wide=prod_wide.sort()
multi=prod_wide.loc['PGE-CSI-00001':'PGE-CSI-00010'].copy()

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
	data=prod_wide, hue="own_sect", scatter=False)
plt.show()

sns.lmplot(x="months_operation", y="prod_index",
	data=prod_wide, hue="program", scatter=False)
plt.show()

sns.lmplot(x="months_operation", y="prod_index",
	data=prod_wide, row="program", col = "own_sect", scatter=False)
plt.show()

#some general descriptives
prod_wide.groupby(["program"]).describe()

#get cross-section of data
prod_wide.reset_index(inplace=True)
cross_section=prod_wide.groupby("app_num").apply(lambda x: x.iloc[0])
del cross_section["app_num"]
cross_section.reset_index(inplace=True)
cross_section=cross_section.rename(columns={'prod_end_date': 'first_prod',
	'prod_kwh': 'avg_prod_kwh'})
cross_section["avg_prod_kwh"]=prod_wide.groupby("app_num")["prod_kwh"].mean()

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
