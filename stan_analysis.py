#stan_analysis.py

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

solar_data = pd.read_csv("/Users/johannesmauritzen/research/solar_files/prod_wide_clean.csv")

solar_data = solar_data[solar_data.prod_index.notnull()]
solar_data = solar_data[solar_data.months_operation.notnull()]

app_nums  = solar_data.app_num.unique()
installation=[i + 1 for i in range(0,len(app_nums))]
installations = pd.DataFrame({"app_num":app_nums, "installation":installation})

solar_data = solar_data.merge(installations, how="left", on="app_num")


#age of panel system in data
N = solar_data.count(axis=0)[0]

app_num = solar_data["app_num"]
unique_app_num = app_num.unique()
J = unique_app_num.size
L = 2

solar_data["log_prod_kwh"] = np.log(solar_data["prod_kwh"])
#sns.lmplot(x="months_operation", y="log_prod_kwh", data = solar_data[solar_data.app_num =="PGE-CSI-00002"])
#lt.show()

#observation level data
log_prod_kwh = solar_data["log_prod_kwh"] 
month = solar_data["month"]

fig, ax = plt.subplots()
plt.scatter

#installation level variables
installations = solar_data[["app_num", "installation", "cust_county", "sector", "tracking", "sys_size_dc", "third_party_owned", "installation_year"]][~solar_data.installation.duplicated()]

#lease variables
pd.get_dummies(solar_data["third_party_owned"])["yes"]
lease = pd.get_dummies(installations.third_party_owned)
lease = [int(i+1) for i in lease.yes]

#Add installation level effects county, sector random effect
counties = installations["cust_county"][~installations.cust_county.duplicated()]
C = len(counties)
county_num = [i + 1 for i in range(0,len(counties))]
county_dict = dict(zip(counties , county_num))
county_var = [county_dict[i] for i in installations.cust_county]

sector = installations["sector"][~installations.sector.duplicated()]
S = len(sector)
sector_num = [i + 1 for i in range(0,len(sector))]
sector_dict = dict(zip(sector, sector_num))
sector_var = [sector_dict[i] for i in installations.sector]

tracking = installations["tracking"][~installations.tracking.duplicated()]
T = len(tracking)
tracking_num = [i + 1 for i in range(0,len(tracking))]
tracking_dict = dict(zip(tracking, tracking_num))
tracking_var = [tracking_dict[i] for i in installations.tracking]


solar_stan_data = {"log_prod_kwh":solar_data["log_prod_kwh"],
"months_operation":solar_data["months_operation"],
"month": month,
"installation":solar_data["installation"],
"sys_size": installations["sys_size_dc"],
"installation_year":installations["installation_year"],
"lease":lease,
"county":county_var,
"tracking":tracking_var,
"sector":sector_var,
"N":N,
"J,":J,
"L":L,
"M":12,
"C":C,
"S":S,
"T":T
}


# def initfun():
# 	return dict(a=.9, b_1=0, sigma=sd_prod_index, nuMinusOne=1) 

# #with init function
# # solar_fit = pystan.stan(model_code=solar_stan,
# #  data = solar_stan_data, iter=500, chains=4,
# #  init=initfun)

# #simple fit
# solar_fit = pystan.stan(file="research/solar_prod/solar_prod.stan",
# 	data = solar_stan_data, iter=1000, chains=4)

# solar_extr = solar_fit.extract(permuted=True) 
#traceplot for single parameter
# solar_fit.plot(["mu_b1"])
# plt.show()

# pickle.dump(solar_extr, open("research/solar_files/solar3.pkl", 'wb')) 
solar_extr = pickle.load(open("/Users/johannesmauritzen/research/solar_files/solar2.pkl", 'rb'))


#prediction
#dataframe of parameter estimates:
solar_params = pd.DataFrame({"mu_b1_own": solar_extr["mu_b1"][:,0],
	"mu_b1_lease":solar_extr["mu_b1"][:,1],
	"sigma":solar_extr["sigma"],
	"jan":solar_extr["mu_mon"][:,0],
	"feb":solar_extr["mu_mon"][:,1],
	"mar":solar_extr["mu_mon"][:,2],
	"apr":solar_extr["mu_mon"][:,3],
	"may":solar_extr["mu_mon"][:,4],
	"jun":solar_extr["mu_mon"][:,5],
	"jul":solar_extr["mu_mon"][:,6],
	"aug":solar_extr["mu_mon"][:,7],
	"sep":solar_extr["mu_mon"][:,8],
	"oct":solar_extr["mu_mon"][:,9],
	"nov":solar_extr["mu_mon"][:,10],
	"dec":solar_extr["mu_mon"][:,11],
	})

cols = ["mu_b1_own", "mu_b1_lease", "sigma", "jan", "feb",
 "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", 
 "nov", "dec"]

solar_params = solar_params[cols]
from stan_helpers import summary_info
summary_table=summary_info(solar_params)
summary_table = summary_table.set_index("params")

#format function
def f1(x):
    return '%1.4f' % x
summary_table.to_latex(buf="table.tex", float_format=f1)

#prediction
def stanPred(p):
	fitted =medParam["jul"] + p*op_time
	return(pd.Series({"fitted":fitted}))

op_time = np.arange(1,61)

#med parameters:
medParam = solar_params[["jul", "mu_b1_own", "mu_b1_lease"]].median()

yhat_own = np.exp(stanPred(medParam["mu_b1_own"])[0])
yhat_own = yhat_own/yhat_own[0]*100

yhat_lease = np.exp(stanPred(medParam["mu_b1_lease"])[0])
yhat_lease = yhat_lease/yhat_lease[0]*100

#predicted values for each chain:
chain_pred_own = solar_params["mu_b1_own"].apply(stanPred)
chain_pred_lease = solar_params["mu_b1_lease"].apply(stanPred)

#create random index for chain sampling
idx = np.random.choice(1999, 100)

fig, ax = plt.subplots()
#plot each chain
for i in range(len(idx)):
	line = np.exp(chain_pred_own.iloc[idx[i],0])
	line = line/line[0]*100
	ax.plot(op_time, line , color="green", alpha=.05, 
		linestyle="-")

for i in range(len(idx)):
	line = np.exp(chain_pred_lease.iloc[idx[i],0])
	line = line/line[0]*100
	ax.plot(op_time, line , color="blue", alpha=.05, 
		linestyle="-")

ax.plot(op_time, yhat_own, color="green", lw = 2)
ax.plot(op_time, yhat_lease, color="blue", lw=2)
ax.set_xlabel("Months of production")
ax.set_ylabel("Predicted production, normalized")
ax.text(50,95, "Host owned")
ax.text(50,98, "Leased")

fig.savefig("research/solar_prod/figures/predicted_deg.png", dpi=150)


#solar_extr = pickle.load(open('research/solar_prod/solar2.pkl', 'rb'))


b1 = solar_extr["b1"]
mu_b1 = solar_extr["mu_b1"]

diff_mu_b1 = mu_b1[:,0] - mu_b1[:,1]

diff_mu_b1 = np.sort(diff_mu_b1)
perc95 = diff_mu_b1[int(round(len(diff_mu_b1)*.95, 0))]

fig, (ax1,ax2) = plt.subplots(2)
ax1.hist(mu_b1[:,0]*12, color="red",alpha=.3, bins=50)
ax1.hist(mu_b1[:,1]*12, color="blue", alpha=.3, bins=50)
ax1.set_xlabel("Posterior distribution of mean deterioration")
ax1.text(-.010, 120,
	r'$\mu_{\beta_1, bought}$',
	fontsize=15)
ax1.text(-.005, 120,
	r'$\mu_{\beta_1, lease}$',
	fontsize=15)
ax2.hist(diff_mu_b1*12, color="red", bins=50)
ax2.axvline(x=perc95)
ax2.text(-.020, 120,
	r'$\mu_{\beta_1, bought} - \mu_{\beta_1, lease}$',
	fontsize=15)
ax2.set_xlabel("Mean difference in deterioration, bought vs. leased")
plt.tight_layout()
fig.savefig("figures/post_mu_b1.png", dpi=150)

#plt.show()

#pickle.dump(solar_extr, open("research/solar_prod/solar1.pkl", 'wb')) 

#open again
#sm = pickle.load(open('research/solar_prod/solar1.pkl', 'rb'))

#solar_fit2 = pystan.stan(fit=solar_fit, data = solar_stan_data,
#	iter=1000, chains=4)
