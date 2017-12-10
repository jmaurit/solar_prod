#data_clean.py
#last updated 2.2.2018
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)

#%%
#import data solar initiative
#Measured production data
#from http://www.californiasolarstatistics.ca.gov/current_data_files/

#"Projects that receive Performance-Based Incentives (PBI)
# also provide monthly system production data, which can be found in
# the Measured Production Data Set below. ""

#%%

def format_production():
    prod=pd.read_csv("/Users/johannesmauritzen/research/solar_prod_data/MeasuredProduction_1-24-2018.csv")


    prod.columns.values

    prod.columns=["app_num", "prog_admin", "program", "tot_cost", "own_sect",
    "host_cust_sect","cust_city", "cust_county", "cust_zip", "pbi_payment_num",
    "prod_end_date", "prod_kwh"]


    prod.head()
    #convert date to date format

    prod["prod_end_date"]=pd.to_datetime(prod["prod_end_date"], format="%Y-%m-%d")

    #convert production from int to float
    prod["prod_kwh"]=prod["prod_kwh"].astype(float)
    return(prod)

#%%
#Link to CSI data
def add_csi_data(prod):
    CSI = pd.read_csv("/Users/johannesmauritzen/research/solar_prod_data/CSI_data.csv")

    inc_columns = ['Application Number', 'Program',
           'Incentive Design', 'Incentive Type', 'Design Factor', 'Incentive Step','CSI Rating',
           'Incentive Amount', 'First Completed Date', 'Total Cost', 'Nameplate Rating','PV Module#1 Manufacturer', 'Is a PBI Buyout Application',
           'Host Customer Sector', 'System Owner Sector', 'Host Customer NAICS Code',
           'Host Customer Physical Address County', 'Solar Contractor Company Name',
           'Contractor License Number', '3rd Party Owner', 'Completed/PBI-In Payment Status']

    new_col_names = ['app_num', 'program',
           'inc_design', 'inc_type', 'design_factor', 'incentive_step','csi_rating',
           'incentive_amount', 'first_completed_date', 'total_cost', 'nameplate_rating','module_manufacturer', 'PBI_Buyout',
           'host_sector', 'owner_sector', 'host_NAICS',
           'county', 'contractor',
           'contractor_num', 'third_party_owned', 'status']

    CSI = CSI[inc_columns]
    CSI.columns = new_col_names


    prod_csi = prod.merge(CSI, on="app_num")
    prod_csi = prod_csi.loc[prod_csi.module_manufacturer.notnull(), :]
    return(prod_csi)
#%%

#%%
#link to interconnected data and format
#I use CSI data not interconnected.
def link_interconnected(prod):
    interconnected = pd.read_csv("/Users/johannesmauritzen/research/solar_prod_data/NEM_CurrentlyInterconnectedDataset_2017-11-30.csv")

    inc_columns = ['Application Id', 'Utility','Service City', 'Service Zip','Service County', 'Technology Type','System Size DC', 'System Size AC','Mounting Method', 'Tracking','Customer Sector', 'App Received Date','App Complete Date','App Approved Date', 'Installer Name', 'Third Party Owned','Third Party Owned Type', 'Third Party Name','Pace Financed', 'Pace Financier','Electric Vehicle',
    'Electric Vehicle Count','Total System Cost','Itc Cost Basis','Matched CSI Application Number', 'Application Status','Project is VNEM, NEM-V, NEM-Agg?', 'Module Model 1','Module Manufacturer 1']


    interconnected = interconnected[inc_columns]

    interconnected.columns = [['app_id', 'utility', 'service_city', 'service_zip','service_county', 'technology','sys_size_dc','sys_size_ac','mounting', 'tracking','sector', 'app_received','app_complete', 'app_approved','contractor','third_party_owned','third_party_owned_type','third_party_name','pace','pace_financier','ev','ev_count','total_system_cost','itc_cost_basis','app_num', 'app_status','Project is VNEM, NEM-V, NEM-Agg?', 'module','module_manufacturer']]

    prod_wide=prod.merge(interconnected, on="app_num")
    prod_wide = prod_wide.loc[prod_wide.module_manufacturer.notnull(), :]
    prod_wide = prod_wide.loc[prod_wide.third_party_owned.notnull(), :]

    return(prod_interconnected)

#prod_inter = link_interconnected(prod)





#%%
def format_prod_csi(prod_csi):
    #%%Format data for csi

    #scale production data
    def standardize(x):
        return((x-np.mean(x))/(2*np.std(x)))

    def make_index(x):
        return((x/np.max(x)))

    prod_csi["prod_scaled"] = prod_csi.groupby("app_num")["prod_kwh"].transform(standardize)

    prod_csi["months_operation"] = prod_csi.pbi_payment_num
    prod_csi["total_months_operation"] = prod_csi.groupby("app_num")["months_operation"].transform(np.max)


    prod_csi["month"] = prod_csi["prod_end_date"].dt.month
    prod_csi["year"] = prod_csi["prod_end_date"].dt.year

    prod_csi["first_prod_year"] = prod_csi.groupby("app_num")["year"].transform(np.min)


    #check for bad data
    #create capacity utilization
    #theoretical max capacity utilization
    #system size in kw, x approx sun hours in a day (12) x approx days in month 30

    #capacity utilization:
    prod_csi["cap_util"] = prod_csi["prod_kwh"]/(prod_csi["csi_rating"]*12*30)
    #get rid of data where cap util is more than 1 or less than .01
    prod_csi = prod_csi.loc[(prod_csi.cap_util<1)&(prod_csi.cap_util>.01)]

    #get rid of systems that have under 2 years of data
    prod_csi = prod_csi.loc[prod_csi.total_months_operation>=24,:]

    #cost per kw
    prod_csi["cost_per_kw"] = prod_csi["total_cost"]/prod_csi["csi_rating"]

    #scale cost per kw
    prod_csi["cost_per_kw_scaled"] = standardize(prod_csi["cost_per_kw"])

    #scale months_operation
    prod_csi["months_operation_scaled"] = standardize(prod_csi["months_operation"])

    #scale first_prod_year
    prod_csi["first_prod_year_scaled"] = standardize(prod_csi["first_prod_year"])

    return(prod_csi)


def main():
    prod = format_production()
    prod_csi = add_csi_data(prod)
    prod_csi = format_prod_csi(prod_csi)
    prod_csi.to_csv("data/prod_csi.csv", index=False)
