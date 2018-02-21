#Stan_results_analysis
#last updated 17.2.2018
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
#plt.rcParams["axes.color_cycle"] = multi



solar_file = open("/Users/johannesmauritzen/research/solar_prod_data/stan_extracts/solar_extr_mac_small.pkl", "rb")
solar_extr = pickle.load(solar_file)

solar_extr.keys()

mu1_leaseDF = pd.DataFrame(solar_extr["mu1_lease"])
beta1_costDF = pd.DataFrame(solar_extr["beta1_cost"])


mu1_leaseDF.head()
beta1_costDF.head()

beta1_costDF.hist()
plt.show()

mu1_leaseDF["mu_diff"] = mu1_leaseDF[1]-mu1_leaseDF[0]
beta1_costDF["beta_diff"] = beta1_costDF[1]-beta1_costDF[0]

beta1_costDF.head()
beta1_costDF.loc[beta1_costDF.beta_diff>0, "beta_diff"].size/2000



fig, ax = plt.subplots(2)
mu1_leaseDF["diff"].hist(ax=ax[0])
beta1_costDF["diff"].hist(ax=ax[1])
ax[0].axvline(x=0, color="red")
ax[1].axvline(x=0, color="red")
fig.set_size_inches(9,9)
ax[0].set_xlim(-.005, .012)
ax[1].set_xlim(-.005, .012)
ax[0].set_xlabel(r"$\mu_{lease,yes}- \mu_{lease,no}$")
ax[1].set_xlabel(r"$\beta_{cost,l=yes}- \beta_{cost,l=no}$")
ax[1].text(.005, 300, "89%>0")
plt.savefig("figures/bayes_hypos.png", dpi=100)
plt.tight_layout()
plt.show()



mu1_lease
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
