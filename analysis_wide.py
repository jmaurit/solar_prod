#analysis_wide.py

import pandas as pd
import tabulate as tab
import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys
import pystan
import scipy
import pickle

pd.options.display.max_rows = 999
pd.options.display.max_columns = 50

#data file with many covariates, but limited sample
prod_wide=pd.read_csv("research/solar_prod/prod_wide_new.csv")
prod_wide["prod_end_date"]=pd.to_datetime(prod_wide["prod_end_date"])
prod_wide["app_received"]=pd.to_datetime(prod_wide["app_received"])
prod_wide["app_complete"]=pd.to_datetime(prod_wide["app_complete"])
prod_wide["app_approved"]=pd.to_datetime(prod_wide["app_approved"])


wide_cross = prod_wide[~prod_wide.app_num.duplicated()]
wide_cross.groupby("program").size()
wide_cross.groupby("own_sect").size()
wide_cross.groupby("host_cust_sect").size()
wide_cross.groupby("cust_county").size()
wide_cross.groupby("technology").size()
wide_cross.groupby("sector").size()
wide_cross.groupby("pace").size()
wide_cross.groupby("ev").size()
wide_cross.groupby("utility").size()
wide_cross.groupby(["third_party_owned", "host_cust_sect"]).size()

#prod_wide.reset_index(inplace=True)
#calculate theoretical maximul
prod_wide["prod_max"]=prod_wide["sys_size_ac"]*12*30.5
prod_wide["prod_index"] = prod_wide["prod_kwh"]/prod_wide["prod_max"]

#create months of operation variable
count_index = lambda v: [i+1 for i, x in enumerate(v)]
prod_wide["months_operation"] = prod_wide.groupby("app_num")["pbi_payment_num"].transform(count_index)

#look for outliers
median_prod_index= prod_wide.groupby("app_num")["prod_index"].apply(lambda x: x.median())
outlier = median_prod_index.index[median_prod_index<.1]
outlier=outlier.get_values()
outlier_data = prod_wide[prod_wide["app_num"].isin(outlier)]

tot_months=lambda x: np.max(x)
prod_wide["tot_months_oper"] = prod_wide.groupby("app_num")["months_operation"].transform(tot_months)

prod_wide = prod_wide.sort(["tot_months_oper", "app_num", "months_operation"], ascending = False)
examples = prod_wide.app_num.unique()[2:4]
example_data = prod_wide[prod_wide["app_num"].isin(examples)]

#plot example
prod_plot = sns.lmplot(x="months_operation", y="prod_index", hue="app_num", data=example_data)
prod_plot.set_xlabels("Months Operation")
prod_plot.set_ylabels("Norm. Production, Max = 1")
prod_plot.fig.set_dpi(150)
prod_plot.fig.set_size_inches(6,4)
#prod_plot.savefig("research/solar_prod/example_prod.png")
plt.show()

prod_wide = prod_wide[~prod_wide["app_num"].isin(outlier)]
prod_wide = prod_wide.sort(["app_num", "prod_end_date"])

#get rid of those with zeros

def is_zero(prod):
	for i in prod:
		if i==0:
			return True
	return False

have_zero_prod=prod_wide.groupby("app_num")["prod_index"].apply(is_zero)
zero_values = have_zero_prod.index[have_zero_prod]
zero_values = zero_values.get_values()

prod_wide = prod_wide[~prod_wide["app_num"].isin(zero_values)]

prod_wide = prod_wide[prod_wide.third_party_owned.notnull()]

prod_wide["lease"]=pd.get_dummies(prod_wide["third_party_owned"])["yes"]

#get rid of outliers
prod_wide=prod_wide[prod_wide.prod_index<1]

prod_wide = prod_wide[prod_wide.tot_months_oper>50]


lease = prod_wide[prod_wide["lease"]==1]
nolease = prod_wide[prod_wide["lease"]==0]

#chart by own_sector
fig, ax = plt.subplots(2)
def plot_prod(data, color, axis):
	ax[axis].plot(data["prod_end_date"], data["prod_index"], alpha=.2, color=color)
lease.groupby("app_num")[["prod_end_date", "prod_index"]].apply(plot_prod, color="blue", axis=0)
nolease.groupby("app_num")[["prod_end_date", "prod_index"]].apply(plot_prod, color="red", axis=1)
ax[0].set_xlabel("Self-owned")
ax[1].set_xlabel("Leased")
fig.tight_layout()
fig.savefig("research/solar_prod/tot_production.png")
plt.show()

prod_wide["month"] = prod_wide["prod_end_date"].dt.month
prod_wide["year"] = prod_wide["prod_end_date"].dt.year

prod_wide["installation_year"] = prod_wide.groupby("app_num")["year"].transform(np.min)



#save for analysis
#prod_wide.to_csv("research/solar_prod/prod_wide_clean.csv")
prod_wide = pd.read_csv("research/solar_prod/prod_wide_clean.csv")
prod_wide["prod_end_date"]= pd.to_datetime(prod_wide["prod_end_date"], format = "%Y-%m-%d")

fe_formula = """prod_index ~ months_operation*lease"""
fe_model = smf.glm(formula=fe_formula, data=prod_wide, 
	family=sm.families.Gaussian(sm.families.links.log)).fit()
fe_model.summary()


plt.subplot2grid(grid_size, (0, 0))

plt.subplot2grid(grid_size, (1, 0))

plt.tight_layout()
plt.show()

#chart by 



