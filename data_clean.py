#data_clean.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)

#import data solar initiative
#Measured production data
#from http://www.californiasolarstatistics.ca.gov/current_data_files/

#"Projects that receive Performance-Based Incentives (PBI) 
# also provide monthly system production data, which can be found in 
# the Measured Production Data Set below. ""
prod=pd.read_csv("research/solar_prod/MeasuredProduction_2-25-2015.csv")

prod.columns=["app_num", "prog_admin", "program", "tot_cost", "own_sect", 
"host_cust_sect","cust_city", "cust_county", "cust_zip", "pbi_payment_num",
"prod_end_date", "prod_kwh"]

#convert date to date format
prod["prod_end_date"]=pd.to_datetime(prod["prod_end_date"], format="%Y-%m-%d")

#convert production from int to float
prod["prod_kwh"]=prod["prod_kwh"].astype(float)


#link to data on installations
installations=pd.read_csv("/Users/johannesmauritzen/Google Drive/Research/CalSolar/multilevel_data.csv")
installations.rename(columns={'Application.Number': 'app_num'}, inplace=True)

interconnected = pd.read_csv("/Users/johannesmauritzen/research/solar_prod/interconnected_solar.csv")
interconnected = interconnected[['Application Id', 'Utility', 
		'Service City', 'Service Zip',
     	 'Service County', 'Technology Type', 'System Size DC',
     	  'System Size AC','Mounting Method', 'Tracking',
       	 'Customer Sector', 'App Received Date', 'App Complete Date', 
       	 'App Approved Date', 'Installer Name', 'Third Party Owned',
       	  'Third Party Owned Type', 'Third Party Name', 
       	  'Pace Financed', 'Pace Financier',
       'Electric Vehicle', 'Electric Vehicle Count', 
       'Total System Cost','Itc Cost Basis', 
       'Matched CSI Application Number', 'Application Status',
       'Project is VNEM, NEM-V, NEM-Agg?', 'Module Model 1',
       'Module Manufacturer 1']]

interconnected.columns = [['app_id', 'utility', 
		'service_city', 'service_zip',
     	 'service_county', 'technology', 'sys_size_dc',
     	  'sys_size_ac','mounting', 'tracking',
       	 'sector', 'app_received', 'app_complete', 
       	 'app_approved', 'contractor', 'third_party_owned',
       	  'third_party_owned_type', 'third_party_name', 
       	  'pace', 'pace_financier',
       'ev', 'ev_count', 
       'total_system_cost','itc_cost_basis',
       'app_num', 'app_status',
       'Project is VNEM, NEM-V, NEM-Agg?', 'module',
       'module_manufacturer']]

#look at one plant
installations=installations[['complete_date','app_num', 
	'incentive_amount', 'nameplate_capacity', 'seller', 
	'manufacturer','contractor', 'cost_per_kw','lease', 
	'latitude','longitude', 'nationality', 'county', 'program', ]]

prod_wide=prod.merge(installations, on="app_num")

prod_wide_new=prod.merge(interconnected, on="app_num")

prod_wide_new.to_csv("research/solar_prod/prod_wide_new.csv", index=False)

prod_wide.to_csv("research/solar_prod/prod_wide.csv", index=False)

prod.to_csv("research/solar_prod/prod_long.csv", index=False)



#other data
#17.7.2015
cal_prod_raw=pd.read_csv("research/solar_prod/MeasuredProduction_2-25-2015.csv")


solar_plants_13 = pd.read_excel("research/power_plants_data/eia8602013/3_3_Solar_Y2013.xlsx", header=1)
  
solar_plants_14 = pd.read_excel("research/power_plants_data/eia8602014er/3_3_Solar_Y2014_Data_Early_Release.xlsx", header=2)




