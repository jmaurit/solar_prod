#data_checks.py
#check which series have positive slopes

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


#check with systems have positive estimated slope
solar_data = pd.read_csv("data/prod_csi.csv")

solar_data.columns.values

solar_data.total_months_operation.head(100)
test_ser = solar_data.loc[solar_data.app_num=="SD-CSI-00001", :]


def slope_est(df):
    #df= test_ser
    res = smf.ols("prod_scaled~months_operation", data=df).fit()
    b = res.params[1]
    return(b)

b_df = solar_data.groupby("app_num").apply(slope_est)


b_df.head()


b_df[b_df>0.01].size

posit_b = b_df.index[b_df>0.04]

posit_df = solar_data.loc[solar_data.app_num.isin(posit_b),:]

grouped_df = posit_df.groupby("app_num")
fig, ax = plt.subplots()
for df in grouped_df:
    ax.plot(df[1].months_operation, df[1].prod_scaled)
plt.show()
