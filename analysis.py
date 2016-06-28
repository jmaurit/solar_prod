#analysis.py

import pandas as pd
import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys
import pystan
import pickle
import scipy

pd.options.display.max_rows = 999
pd.options.display.max_columns = 50

solar_data= pd.read_csv("research/solar_prod/solar_data.csv")

#get rid of NAs

solar_data = solar_data[solar_data.prod_index.notnull()]
solar_data = solar_data[solar_data.months_operation.notnull()]

N = solar_data.count(axis=0)[0]

mean_prod_index = solar_data["prod_index"].mean()
mean_months_operation = solar_data["months_operation"].mean()
sd_prod_index = solar_data["prod_index"].std()
sd_months_operation = solar_data["months_operation"].std()


solar_stan_data = {"prod_index":solar_data["prod_index"],
"months_operation":solar_data["months_operation"],
"N":N, 
"mean_prod_index":mean_prod_index,
"mean_months_operation":mean_months_operation,
"sd_prod_index":sd_prod_index,
"sd_months_operation":sd_months_operation
}

solar_stan = """
data {
	int<lower = 0> N; // number of observations
	vector[N] prod_index; //response variable
	vector[N] months_operation; //main predictor
	real mean_prod_index;
	real mean_months_operation;
	real sd_prod_index;
	real sd_months_operation;
}
transformed data{ //for constants in priors
	real unifLo;
	real unifHi;
	real expLambda;
	real alphasigma;
	real beta1sigma;
	unifLo <- sd_prod_index/1000;
	unifHi <- sd_prod_index*1000;
	expLambda<-1/29.0;
	alphasigma <- 10*fabs(mean_months_operation*sd_prod_index/sd_months_operation);
	beta1sigma <- 10*fabs(sd_prod_index/sd_months_operation);
}
parameters {
	real a; //intercept
	real b_1; //slope
	real<lower=0> sigma; //standard deviation
	real<lower=0> nuMinusOne;
}
transformed parameters {
	real<lower=0> nu;
	nu<-nuMinusOne+1;
}
model {
	sigma ~ uniform(unifLo, unifHi);
	nuMinusOne ~ exponential(expLambda);
	a ~ normal(0, alphasigma);
	b_1 ~ normal(0, beta1sigma);

	prod_index~student_t(nu, 
		a + b_1*months_operation, 
		sigma);
}
"""

def initfun():
	return dict(a=.9, b_1=0, sigma=sd_prod_index, nuMinusOne=1) 

solar_fit = pystan.stan(model_code=solar_stan,
 data = solar_stan_data, iter=500, chains=4,
 init=initfun)

#pickle
solar_extr = solar_fit.extract() 
pickle.dump(solar_extr, open("research/solar_prod/solar1.pkl", 'wb')) 

#open again
sm = pickle.load(open('research/solar_prod/solar1.pkl', 'rb'))

#solar_fit2 = pystan.stan(fit=solar_fit, data = solar_stan_data,
#	iter=1000, chains=4)



print(solar_fit)
solar_fit.plot()
plt.show()


solar_results = solar_fit.extract(permuted=True)
b_1 = solar_results["b_1"]




