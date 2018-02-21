#stan_analysis_new.py
#stan MCMC analysis from 2.6.2017

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



#get_ipython().magic('matplotlib inline')


def get_percentiles(ser):
    return(np.percentile(ser, [2.5,15,50,85,97.5]))

def logit_to_prob(value):
    return(math.exp(value)/(1+math.exp(value)))

def load_data():
    solar_data = pd.read_csv("data/prod_csi.csv")
    return(solar_data)

def format_data(solar_data):
    solar_data = solar_data[solar_data.prod_scaled.notnull()]
    solar_data = solar_data[solar_data.months_operation_scaled.notnull()]
    #limit to data over three years
    solar_data = solar_data[solar_data.total_months_operation>=36]
    return(solar_data)

def descale_data(solar_series):
    return((solar_series - np.mean(solar_series))/(2*np.std(solar_series)))

def transform_log(solar_data):
    solar_data.loc[:,"log_prod_kwh"] = np.log(solar_data.prod_kwh+ 0.00001)
    solar_data.loc[:,"log_cost_per_kw"] = np.log(solar_data.cost_per_kw + 0.00001)
    solar_data.loc[:,"log_csi_rating"] = np.log(solar_data.csi_rating + 0.00001)
    return(solar_data)

def create_num_dict(sseries):
    skeys  = sseries.unique()
    #numbered from 1 to N
    svalues=[i + 1 for i in range(0,len(skeys))]
    #create dict with both
    sys_dict = dict(zip(skeys, svalues))
    return(sys_dict)

def create_stan_data(solar_data):
    """
    Takes in dataframe and returns dict of stan data
    """
    #Observation level data (size N)
    #age of panel system in data
    N = solar_data.count(axis=0)[0]
    app_num = solar_data["app_num"]
    log_prod_kwh = solar_data["log_prod_kwh"]
    month = solar_data["month"]
    months_operation = solar_data["months_operation"]

    #create system_id from 1 to J for N data points
    sys_dict = create_num_dict(solar_data["app_num"])
    system_id =app_num.map(sys_dict)

    system_level_df = solar_data[["app_num", "owner_sector", "log_csi_rating", "third_party_owned","module_manufacturer", "first_prod_year", "incentive_step", "log_cost_per_kw"]][~solar_data.app_num.duplicated()]

    unique_system_id = system_level_df["app_num"].map(sys_dict)
    J = unique_system_id.size

    #continuous variables at system level
    log_csi_rating = system_level_df["log_csi_rating"]
    log_cost_per_kw = system_level_df["log_cost_per_kw"]
    first_prod_year = system_level_df["first_prod_year"]
    incentive_step = system_level_df["incentive_step"]

    #solar_data["incentive_step"].unique()

    #module_manufacturer data
    module_dict = create_num_dict(system_level_df["module_manufacturer"])
    module_id = system_level_df["module_manufacturer"].map(module_dict)
    M = module_id.unique().size

    #sector categories
    owner_sect_dict = create_num_dict(system_level_df["owner_sector"])
    #for varying effects
    owner_sect_id = system_level_df["owner_sector"].map(owner_sect_dict)
    #for dummy variables
    owner_sect_dummies = pd.get_dummies(system_level_df.owner_sector)
    S=4

    #lease variables
    L = 2
    lease_dict = create_num_dict(system_level_df["third_party_owned"])
    leased = system_level_df["third_party_owned"].map(lease_dict)

    solar_stan_data={
    "N":N, #observations
    "J":J, #systems
    "M":M, #module Manufacturers
    "L":L, #leased or not
    "S":S,
    #observation data
    "system_id":system_id,
    "log_prod_kwh": log_prod_kwh,
    "months_operation": months_operation,
    "month": month,

    #system level data,
    "log_cost_per_kw":log_cost_per_kw,
    "log_csi_rating": log_csi_rating,
    "first_prod_year": first_prod_year,
    #"incentive_step": incentive_step,
    "module_id": module_id,
    "owner_sect_id": owner_sect_id,
    "leased":leased
    }
    return(solar_stan_data)

def solar_stan_model():
    solar_stan_model1 = """
    data{
    	int<lower = 0> N; //number of observations
    	int<lower = 0> J; //number of groups
        int<lower = 0> M; //number of manufacturers
    	int<lower = 0> L; //lease/not lease =2
        int<lower=0> S; //Number of sectors

        //Observation Data
    	int system_id[N]; //Which of J installations does it belong too
    	vector[N] log_prod_kwh; //response variable
    	vector[N] months_operation; //main predictor
    	int month[N]; // which of 12 months does it belong too.

    	int leased[J]; //indicator variable for lease
        vector[J] log_cost_per_kw; //cost per kw log
        vector[J] log_csi_rating; //size of system
        vector[J] first_prod_year; //year of first operation
        //vector[J] incentive_step;

        int module_id[J];
        int owner_sect_id[J];
    }

    transformed data{
        vector[N] st_log_prod_kwh;
        vector[N] st_months_operation;
        vector[J] st_log_cost_per_kw;
        vector[J] st_log_csi_rating;
        //vector[J] st_incentive_step;
        vector[J] st_first_prod_year;

        st_log_prod_kwh = (log_prod_kwh - mean(log_prod_kwh))/(2*sd(log_prod_kwh));

        st_months_operation = (months_operation - mean(months_operation))/(2*sd(months_operation));

        st_log_csi_rating = (log_csi_rating - mean(log_csi_rating))/(2*sd(log_csi_rating));

        //st_incentive_step = (incentive_step - mean(incentive_step))/(2*sd(incentive_step));

        st_first_prod_year = (first_prod_year - mean(first_prod_year))/(2*sd(first_prod_year));

        st_log_cost_per_kw = (log_cost_per_kw-mean(log_cost_per_kw))/(2*sd(log_cost_per_kw));


    }

    parameters{
        real meta_mu;
        real meta_beta;
        real meta_mu_mon;

        real<lower=0> sigma_mu1;
        //real<lower=0> sigma_b1;

        real mu_b0;
        //real mu_b1;
        real<lower=0> sigma_b0;
        real<lower=0> sigma_mon;
        real<lower=0> sigma_beta;
        real<lower=0> sigma;

        real b0[J];
        //real b1_re[J];

        real mu_mon[12];

        real mu1_lease[L];
        real mu1_m[M];
        real mu1_s[S];

        real beta1_cost[L];

        real beta1_size;
        real beta1_fy;
    }

    transformed parameters {
    	real b1[J]; // varying slopes by group
    	real y_hat[N]; //individual means

        //system level regression for slopes
    	//for (j in 1:J){
    	  //b1[j]=mu1_lease[leased[j]] +
          //beta1_cost[leased[j]]*st_log_cost_per_kw[j] +
          //mu1_m[module_id[j]] +
          //mu1_s[owner_sect_id[j]] +
          //beta1_size*st_log_csi_rating[j] +
          //beta1_fy*st_first_prod_year[j] +
          //b1_re[j];
    	//}

        for (j in 1:J){
    		b1[j]=mu1_lease[leased[j]] + mu1_s[owner_sect_id[j]]  +
            mu1_m[module_id[j]] + beta1_fy*st_first_prod_year[j] +
            beta1_size*st_log_csi_rating[j] +
            beta1_cost[leased[j]]*st_log_cost_per_kw[j];
            }

    	for (i in 1:N){
    		y_hat[i] = b0[system_id[i]] + b1[system_id[i]]*st_months_operation[i] + mu_mon[month[i]];
    	}
    }

    model{
        //highest level meta variables
        meta_mu ~ cauchy(0,5);
        meta_beta ~ cauchy(0,5);
        sigma_beta ~ cauchy(0,5);

        meta_mu_mon ~ cauchy(0,5);
        sigma_mon ~ cauchy(0,5);

        //meta parameters on system level coefficients.
        sigma_mu1 ~ cauchy(0, 5);
      	//sigma_b1 ~ cauchy(0, 5);

        mu_b0 ~ cauchy(0,5);
        sigma_b0 ~ cauchy(0,5);

    	sigma ~ cauchy(0,5);

        to_vector(b0) ~ normal(mu_b0, sigma_b0);

        //delete these when adding b1 earlier
        //mu_b1 ~ cauchy(0, 5);
        //to_vector(b1_re) ~ normal(mu_b1, sigma_b1);

        to_vector(mu_mon) ~ normal(meta_mu_mon, sigma_mon);

        to_vector(mu1_lease)~ cauchy(meta_mu, sigma_mu1);
        to_vector(mu1_m)~ cauchy(meta_mu, sigma_mu1);
        to_vector(mu1_s)~ cauchy(meta_mu, sigma_mu1);

        to_vector(beta1_cost)~cauchy(0, 5);

        beta1_size~cauchy(meta_beta, sigma_beta);
        beta1_fy~cauchy(meta_beta, sigma_beta);

    	st_log_prod_kwh ~ normal(y_hat, sigma);
    }
    """
    return(solar_stan_model1)

def solar_stan_model2():
    """
    Try to optimize:
    From stan reference manual 9.13
    """
    solar_stan_model1 = """
    data{
    	int<lower = 0> N; //number of observations
    	int<lower = 0> J; //number of groups
        int<lower = 0> M; //number of manufacturers
    	int<lower = 0> L; //lease/not lease =2
        int<lower=0> S; //Number of sectors

        //Observation Data
    	int system_id[N]; //Which of J installations does it belong too
    	vector[N] log_prod_kwh; //response variable
    	vector[N] months_operation; //main predictor
    	int month[N]; // which of 12 months does it belong too.

    	int leased[J]; //indicator variable for lease
        vector[J] log_cost_per_kw; //cost per kw log
        vector[J] log_csi_rating; //size of system
        vector[J] first_prod_year; //year of first operation
        //vector[J] incentive_step;

        int module_id[J];
        int owner_sect_id[J];
    }

    transformed data{
        vector[N] st_log_prod_kwh;
        vector[N] st_months_operation;
        vector[J] st_log_cost_per_kw;
        vector[J] st_log_csi_rating;
        //vector[J] st_incentive_step;
        vector[J] st_first_prod_year;

        st_log_prod_kwh = (log_prod_kwh - mean(log_prod_kwh))/(2*sd(log_prod_kwh));

        st_months_operation = (months_operation - mean(months_operation))/(2*sd(months_operation));

        st_log_csi_rating = (log_csi_rating - mean(log_csi_rating))/(2*sd(log_csi_rating));

        //st_incentive_step = (incentive_step - mean(incentive_step))/(2*sd(incentive_step));

        st_first_prod_year = (first_prod_year - mean(first_prod_year))/(2*sd(first_prod_year));

        st_log_cost_per_kw = (log_cost_per_kw-mean(log_cost_per_kw))/(2*sd(log_cost_per_kw));


    }

    parameters{
        real meta_mu;
        real meta_beta;
        real meta_mu_mon;

        real<lower=0> sigma_mu1;
        //real<lower=0> sigma_b1;

        real mu_b0;
        //real mu_b1;
        real<lower=0> sigma_b0;
        real<lower=0> sigma_mon;
        real<lower=0> sigma_beta;
        real<lower=0> sigma;

        real b0[J];
        //real b1_re[J];

        real mu_mon[12];

        real mu1_lease[L];
        real mu1_m[M];
        real mu1_s[S];

        real beta1_cost[L];

        real beta1_size;
        real beta1_fy;
    }

    transformed parameters {
    	vector[J] b1; // varying slopes by group
    	vector[N] y_hat; //individual means

        //system level regression for slopes
    	//for (j in 1:J){
    	  //b1[j]=mu1_lease[leased[j]] +
          //beta1_cost[leased[j]]*st_log_cost_per_kw[j] +
          //mu1_m[module_id[j]] +
          //mu1_s[owner_sect_id[j]] +
          //beta1_size*st_log_csi_rating[j] +
          //beta1_fy*st_first_prod_year[j] +
          //b1_re[j];
    	//}

        for (j in 1:J){
    		b1[j]=mu1_lease[leased[j]] + mu1_s[owner_sect_id[j]]  +
            mu1_m[module_id[j]] + beta1_fy*st_first_prod_year[j] +
            beta1_size*st_log_csi_rating[j] +
            beta1_cost[leased[j]]*st_log_cost_per_kw[j];
            }

    	for (i in 1:N){
    		y_hat[i] = b0[system_id[i]] + b1[system_id[i]]*st_months_operation[i] + mu_mon[month[i]];
    	}
    }

    model{
        //highest level meta variables
        meta_mu ~ cauchy(0,5);
        meta_beta ~ cauchy(0,5);
        sigma_beta ~ cauchy(0,5);

        meta_mu_mon ~ cauchy(0,5);
        sigma_mon ~ cauchy(0,5);

        //meta parameters on system level coefficients.
        sigma_mu1 ~ cauchy(0, 5);
      	//sigma_b1 ~ cauchy(0, 5);

        mu_b0 ~ cauchy(0,5);
        sigma_b0 ~ cauchy(0,5);

    	sigma ~ cauchy(0,5);

        to_vector(b0) ~ normal(mu_b0, sigma_b0);

        //delete these when adding b1 earlier
        //mu_b1 ~ cauchy(0, 5);
        //to_vector(b1_re) ~ normal(mu_b1, sigma_b1);

        to_vector(mu_mon) ~ normal(meta_mu_mon, sigma_mon);

        to_vector(mu1_lease)~ cauchy(meta_mu, sigma_mu1);
        to_vector(mu1_m)~ cauchy(meta_mu, sigma_mu1);
        to_vector(mu1_s)~ cauchy(meta_mu, sigma_mu1);

        to_vector(beta1_cost)~cauchy(0, 5);

        beta1_size~cauchy(meta_beta, sigma_beta);
        beta1_fy~cauchy(meta_beta, sigma_beta);

    	st_log_prod_kwh ~ normal(y_hat, sigma);
    }
    """
    return(solar_stan_model1)


#run:
solar_data = load_data()
solar_data = format_data(solar_data)
solar_data = transform_log(solar_data)

np.std(solar_data["months_operation"])

.0149/16*12*10


solar_data[["log_prod_kwh", "months_operation", "month"]].apply(np.mean, axis=0)

solar_stan_data = create_stan_data(solar_data);

solar_stan_data.keys()

solar_stan_data["N"]
solar_stan_data["J"]
solar_stan_data["M"]


stan_model_code = solar_stan_model()
solar_stan_model = pystan.StanModel(model_code=stan_model_code)
solar_stan_fit = solar_stan_model.sampling(data=solar_stan_data, iter=1000, chains=4)

solar_extr = solar_stan_fit.extract(permuted=True)

solar_extr.keys()

rel_keys = ('meta_mu', 'meta_beta', 'meta_mu_mon', 'sigma_mu1', 'mu_b0', 'sigma_b0', 'sigma_mon', 'sigma_beta', 'sigma', 'b0', 'mu_mon', 'mu1_lease', 'mu1_m', 'mu1_s', 'beta1_cost', 'beta1_size', 'beta1_fy', 'b1')

solar_extr_small = {k: solar_extr[k] for k in rel_keys}

pickle.dump(solar_extr_small, open("/Users/johannesmauritzen/research/solar_prod_data/stan_extracts/solar_extr_mac_small.pkl", 'wb'), protocol=4)

yhats = solar_extr["y_hat"]
pickle.dump(yhats, open("/Users/johannesmauritzen/research/solar_prod_data/stan_extracts/yhats.pkl", 'wb'), protocol=4)
