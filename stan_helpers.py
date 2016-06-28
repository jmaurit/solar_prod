import pandas as pd
import numpy as np

def get_cent_interval(sample, conf=.95):
	p=1-conf
	sample = np.sort(sample)
	n_sample = len(sample)
	low_ci = sample[round(n_sample*(p/2))]
	high_ci = sample[round(n_sample*(1-p/2))]
	return([low_ci, high_ci])

def summary_info(post_sample, lim=5):	
	#post_sample = solar_params
	#lim=5
	"""
	Input extracted STAN model sample: post_sample
	lim - confidence interval ie lim = 5 => 95 central posterior interval  
	"""

	params = []
	means = []
	medians = []
	modes = []
	low_ci = []
	high_ci = []
	for i in post_sample:
		param_sim = post_sample[i]
		if (param_sim.shape[0] == param_sim.size):
			params.append(i)
			means.append(np.mean(param_sim))
			medians.append(np.median(param_sim))
			n_samples = len(param_sim)
			n_tail = n_samples*(lim/100)/2
			param_sim=np.sort(param_sim)
			low_ci.append(param_sim[n_tail])
			high_ci.append(param_sim[n_samples-n_tail])
		else:	
			for par_case in param_sim.T:
				params.append(i)
				means.append(np.mean(par_case))
				medians.append(np.median(par_case))
				modes.append(np.mode(parcase))
				n_samples = len(par_case)
				n_tail = n_samples*(lim/100)/2
				par_case=np.sort(par_case)
				low_ci.append(par_case[n_tail])
				high_ci.append(par_case[n_samples-n_tail])
	output = {"params":params,
		"means" : means,
		"medians" : medians,
		"low_ci" : low_ci,
		"high_ci" : high_ci}
	output = pd.DataFrame(output)
	return(output)
