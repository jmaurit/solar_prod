
//solar_prod.stan
data{
	int<lower = 0> N; // number of observations
	int<lower = 0> J; //number of groups
	int<lower = 0> L; //lease/not lease =2
	int<lower = 0> M; //number of months
	int installation[N];
	vector[N] log_prod_kwh; //response variable
	vector[N] months_operation; //main predictor
	int month[N]; //months
	int lease[J]; //indicator variable
}

parameters{
	vector[J] b0; //intercepts for ind-level - random effects
	real<lower=0> sigma; //standard deviation
	real<lower=0> sigma_b0; //st.dev group level intercept
	real<lower=0> sigma_b1; //st. dev group level lease
	real mu_b0;	
	real<lower=-1, upper=0> mu_b1[2]; //varying slope, grouped by lease
	real re_b1[J];
	real mu_mon[M];
	real b2;
}

transformed parameters {
	real b1[Y]; // varying slopes by year
	real y_hat[N]; //individual means

	for (j in 1:J){
		b1[j]<-mu_b1[lease[j]] + re_b1[j];
	}
	
	for (i in 1:N){
		y_hat[i] <- b0[installation[i]] + b1[installation[i]]*months_operation[i] + mu_mon[month[i]];
	}
	
}

model{	
  	mu_b1 ~ normal(0, 100); //vectorized,l 
  	sigma_b1 ~ uniform(0, 100);

  	sigma_b0 ~ uniform(0, 100); // mu and sigma on b0 param.
  	//mu_b0 ~ normal(0, 100);

  	re_b1 ~ normal(0, sigma_b1); //vectorized, j 
	b0 ~ normal(mu_b0, sigma_b0); // vectorized, j
	mu_mon ~ normal(1, 10); // vectorized, m
	sigma ~ uniform(0,100);

	log_prod_kwh ~ normal(y_hat, sigma); 
}




