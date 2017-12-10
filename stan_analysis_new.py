#stan_analysis_new.py
#stan MCMC analysis from 2.5.2017


# coding: utf-8

# In[2]:

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


# In[3]:

pd.options.display.max_rows = 999
pd.options.display.max_columns = 50
#get_ipython().magic('matplotlib inline')


# In[4]:

def get_percentiles(ser):
    return(np.percentile(ser, [2.5,15,50,85,97.5]))

def logit_to_prob(value):
    return(math.exp(value)/(1+math.exp(value)))


# In[5]:

solar_data = pd.read_csv("/Users/johannesmauritzen/research/solar_files/prod_csi.csv")

solar_data = solar_data[solar_data.prod_index.notnull()]
solar_data = solar_data[solar_data.months_operation.notnull()]

app_nums  = solar_data.app_num.unique()
installation=[i + 1 for i in range(0,len(app_nums))]
installations = pd.DataFrame({"app_num":app_nums, "installation":installation})
solar_data = solar_data.merge(installations, how="left", on="app_num")


# In[6]:

#age of panel system in data
N = solar_data.count(axis=0)[0]

app_num = solar_data["app_num"]
unique_app_num = app_num.unique()
J = unique_app_num.size
L = 2

solar_data["log_prod_kwh"] = np.log(solar_data["prod_kwh"])


# In[7]:

#observation data
log_prod_kwh = solar_data["log_prod_kwh"]
month = solar_data["month"]


# In[8]:

#installation level variables
installations = solar_data[["app_num", "installation", "cust_county", "sector", "tracking", "sys_size_dc", "third_party_owned", "installation_year"]][~solar_data.installation.duplicated()]


# In[9]:

#lease variables
pd.get_dummies(solar_data["third_party_owned"])["yes"]
lease = pd.get_dummies(installations.third_party_owned)
lease = [int(i+1) for i in lease.yes]


# In[10]:

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


# In[ ]:




# In[11]:

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
"J":J,
"L":L,
"M":12,
"C":C,
"S":S,
"T":T
}


# In[12]:

# def initfun():
# 	return dict(a=.9, b_1=0, sigma=sd_prod_index, nuMinusOne=1)


# In[19]:

solar_stan = """

data{
	int<lower = 0> N; // number of observations
	int<lower = 0> J; //number of groups
	int<lower = 0> L; //lease/not lease =2
	int<lower = 0> M; //number of months
    int<lower = 0> C; //number of different counties
    int<lower = 0> S; //number of different sectors
    int<lower = 0> T; //number of different trackers

	int installation[N]; //Which of J installations does it belong too
	vector[N] log_prod_kwh; //response variable
	vector[N] months_operation; //main predictor
	int month[N]; // which of 12 months does it belong too.

	int lease[J]; //indicator variable for lease
    int county[J];
    int tracking[J];
    int sector[J];

    vector[J] sys_size; //size of system
    vector[J] installation_year; //year of installation

}

transformed data{
    vector[N] st_log_prod_kwh;
    vector[N] st_months_operation;

    vector[J] st_sys_size;
    vector[J] st_installation_year;

    st_log_prod_kwh <- (log_prod_kwh - mean(log_prod_kwh))/sd(log_prod_kwh);
    st_months_operation <- (months_operation - mean(months_operation))/sd(months_operation);
    st_sys_size <- (sys_size - mean(sys_size))/sd(sys_size);
    st_installation_year <- (installation_year - mean(installation_year))/sd(installation_year);
}

parameters{
    real  mu_b0;
	real<upper =0> mu_lease[2]; //varying slope, grouped by lease
	real re_b1[J];
    real re_b0[J];
	real mu_mon[M];
    real beta0_size;
    real beta1_size;
    real beta1_ins_year;

    real mu1_c[C];
    real mu1_s[S];
    real mu1_t[T];

    real mu0_c[C];
    real mu0_t[T];

	real<lower=0> sigma; //standard deviation
	real<lower=0> sigma_b0; //st.dev group level intercept
	real<lower=0> sigma_b1; //st. dev group level lease
    real<lower=0> sigma_mu_lease; //st. dev, mu

    real<lower=0> sigma_mu0_c; //st. dev, mu
    real<lower=0> sigma_mu0_t; //st. dev, mu

    real<lower=0> sigma_mu1_c; //st. dev, mu
    real<lower=0> sigma_mu1_s; //st. dev, mu
    real<lower=0> sigma_mu1_t; //st. dev, mu

    real<lower=0> sigma_mon; //

}

transformed parameters {
	real b1[J]; // varying slopes by group
    real b0[J]; // varying intercept by group
	real y_hat[N]; //individual means

	for (j in 1:J){
		b1[j]<-mu_lease[lease[j]] + mu1_s[sector[j]] + mu1_c[county[j]] + mu1_t[tracking[j]] + beta1_ins_year*st_installation_year[j] + re_b1[j];
	}

    for (j in 1:J){
        b0[j]<-beta0_size*st_sys_size[j] + mu0_c[county[j]] + mu0_t[tracking[j]] + re_b0[j];
    }

	for (i in 1:N){
		y_hat[i] <- b0[installation[i]] + b1[installation[i]]*st_months_operation[i] + mu_mon[month[i]];
	}
}

model{
    sigma_mu_lease ~ cauchy(0,5);
    sigma_mu0_c ~ cauchy(0, 5);
    sigma_mu0_t ~ cauchy(0, 5);

    sigma_mu1_c ~ cauchy(0, 5);
    sigma_mu1_s ~ cauchy(0, 5);
    sigma_mu1_t ~ cauchy(0, 5);

  	sigma_b1 ~ cauchy(0, 5);
  	sigma_b0 ~ cauchy(0, 5); // mu and sigma on b0 param.

    sigma_mon ~ cauchy(0,5);
	sigma ~ cauchy(0,5);

    to_vector(re_b0) ~ cauchy(0, sigma_b0); //vectorized, j
  	to_vector(re_b1) ~ cauchy(0, sigma_b1); //vectorized, j
	to_vector(mu_mon) ~ cauchy(0,sigma_mon); // vectorized, m
    to_vector(mu_lease) ~ cauchy(0, sigma_mu_lease); //vectorized,l

    beta0_size~cauchy(0, 5);
    beta1_size~cauchy(0, 5);
    beta1_ins_year~cauchy(0, 5);

    to_vector(mu1_c)~ cauchy(0, sigma_mu1_c);
    to_vector(mu1_s)~ cauchy(0, sigma_mu1_s);
    to_vector(mu1_t)~ cauchy(0, sigma_mu1_t);

    to_vector(mu0_c)~ cauchy(0, sigma_mu0_c);
    to_vector(mu0_t)~ cauchy(0, sigma_mu0_t);

	st_log_prod_kwh ~ cauchy(y_hat, sigma);
}
"""


solar_fit = pystan.stan(model_code=solar_stan,
            data = solar_stan_data, iter=1000, chains=4)

solar_extr = solar_fit.extract(permuted=True)

pickle.dump(solar_extr, open("/Users/johannesmauritzen/research/solar_files/solar6.pkl", 'wb'))

solar_fit.plot(["mu_lease"])



import pickle
pp_extr = pickle.load(open("/Users/johannesmauritzen/research/solar_files/solar4.pkl", 'rb'))


#traceplot for single parameter
solar_fit.plot(["mu_b1"])
plt.show()
