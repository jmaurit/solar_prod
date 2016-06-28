
//solar_prod_simp.stan

data{
	int<lower = 0> N; // number of observations
	vector[N] prod_index; //response variable
	vector[N] months_operation; //main predictor
	int<lower=1, upper=2> lease[N]; //indicator variable
	real mean_prod_index;
	real mean_months_operation;
	real sd_prod_index;
	real sd_months_operation;
}


transformed data{ //for constants in priors
	real unifLo;
	real unifHi;
	unifLo <- sd_prod_index/1000;
	unifHi <- sd_prod_index*1000;
}


parameters{
	real b0; //intercepts for ind-level
	vector[2] b1; //slopes or ind-level
	real<lower=0> sigma; //standard deviation
	real<lower=0> sigma_b0; //st.dev group level intercept
	real<lower=0> sigma_b1; //st. dev group level lease
	real mu_b0;
}


transformed parameters{
vector[N] y_hat;

for (i in 1:N)
	y_hat[i] <- b0 + b1[lease[i]]*months_operation[i];

//for student_t
//	real<lower=0> nu;
//	nu<-nuMinusOne+1; 
}

model{
	sigma ~ uniform(unifLo, unifHi);
	//nuMinusOne ~ exponential(expLambda); //for student_t
	//at b_months for each month 
	
  	sigma_b1 ~ uniform(0, 100);
  	sigma_b0 ~ uniform(0, 100); // mu and sigma on b0 param.
  	mu_b0 ~ normal(0, 10); 

  	mu_b1 ~ normal(1, 10); //vectorized
	b1 ~ normal(0, 100); //installation-level intercept
	b0 ~ normal(mu_b0, sigma_b0);

  	//can make student t later student_t(nu, a+b1x, sigma)
	prod_index ~ normal(y_hat, sigma); 

}
