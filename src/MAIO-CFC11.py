# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:03:42 2023

@author: rens_
"""
import netCDF4 as nc
import pandas as pd
import xarray as xr
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import os

def fill(dataframe):
    imputer = KNNImputer(n_neighbors = 5, weights = "distance")
    ubfill = pd.DataFrame(imputer.fit_transform(dataframe).ravel())
    ubfill.columns = dataframe.columns
    ubfill.index = dataframe.index
    
    return ubfill

def strip_Ps(dataframe):
    dataframe["CFC-11S"] = dataframe["CFC-11S"].apply(lambda x: x.rstrip("P"))
    dataframe["CFC-11P"] = dataframe["CFC-11P"].apply(lambda x: x.rstrip("P"))
    dataframe["CFC-11S"] = dataframe["CFC-11S"].astype(float)
    dataframe["CFC-11P"] = dataframe["CFC-11P"].astype(float)
    
    return dataframe

def read_csv_data(name):
    x=pd.read_csv(name)
    x['date']=pd.to_datetime(dict(year=x.YYYY, month=x.MM, day=x.DD, hour=x.hh))
    x.set_index('date',inplace=True)
    
    return x[["CFC-11S", "CFC-11P"]]

DATA_PATH = os.path.join(os.curdir, "..\\data")
macehead_1990_file = os.path.join(DATA_PATH, "MHDgage_1990.csv")
macehead_file = os.path.join(DATA_PATH, "AGAGE-GCMD_MHD_cfc-11.nc")
adrigole_file = os.path.join(DATA_PATH, "ADRale_1980.csv")

df_1980 = read_csv_data(adrigole_file)
df_1980 = strip_Ps(df_1980)
df_1980["CFC-11"] = df_1980["CFC-11P"] + df_1980["CFC-11S"]
df_1980 = df_1980.drop(df_1980["CFC-11"][df_1980["CFC-11"] == 0].index)

df_1990 = read_csv_data(macehead_1990_file)
df_1990 = strip_Ps(df_1990)
df_1990 = df_1990.drop(df_1990["CFC-11P"][df_1990["CFC-11P"] == 0].index)

macehead_data = nc.Dataset(macehead_file)
data_dict = {}
for var in macehead_data.variables:
    if 'time' in macehead_data.variables[var].dimensions:
        data_dict[var] = macehead_data.variables[var][:]
macehead_df = pd.DataFrame(data_dict)
macehead_df.iloc[:, 0] = pd.to_datetime(macehead_df.iloc[:, 0], unit='s')
years_to_keep = [2000, 2010, 2021]
macehead_df = macehead_df[macehead_df.iloc[:, 0].dt.year.isin(years_to_keep)]
macehead_df = macehead_df.iloc[:, :4]
