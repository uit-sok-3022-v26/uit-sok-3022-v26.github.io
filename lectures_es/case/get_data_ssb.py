import numpy as np
import pandas as pd
import ssb
import credentials as cr
import pymysql
import paneltime as pt
import os
from statsmodels import api as sm
from matplotlib import pyplot as plt


# import data from titlon
#	* Stock price index to predict inflation
#	* Bonds, as reference point

# Use inflation data from ssb
# Use paneltime to predict

FLDR = os.path.dirname(__file__)
DATAFILE = f'{FLDR}/output/pred_nor_data.dmp'
BONDFILE = f'{FLDR}/data/spbonds_nor.xls'

def get_data():
	if os.path.exists(DATAFILE) and False:
		dfd = pd.read_pickle(DATAFILE)
		return dfd
	bnp = get_bnp()
	kpi = get_kpi()
	bonds = get_bonds()
	indx = get_titlon()
	money = get_money()
	df = pd.concat((kpi, bonds, indx, bnp, money), axis=1).dropna()
	df = df[['KPI',  'svr_bond_index_norw',  'OSEBXLinked',   'BNP', 'M2']]

	dfd = np.log(df)
	dfd = dfd.diff()
	
	quarter = np.array((dfd.index.month-1)/3, dtype=int)+1
	
	df_dum = pd.get_dummies(quarter, 
						 drop_first = True, 
						 prefix = 'Q').set_index(dfd.index)
	dfd = pd.concat((dfd,df_dum), axis = 1)

	dfd['time'] = (dfd.index.year-dfd.index.year[0])*12 + dfd.index.month
	
	dfd.to_pickle(DATAFILE)

	return dfd


def get_kpi():
	df = ssb.kpi()[['Tid', 'Data']]
	df = df.rename(columns={'Data':'KPI', 'Tid':'Date'})
	df['Date'] =  pd.to_datetime(df['Date'], format='%YM%m')
	df = df.set_index('Date')
	return df

def get_bnp():
	df = ssb.bnp()[['Tid', 'Data']]
	df = df.rename(columns={'Data':'BNP', 'Tid':'Date'})
	df['Date'] =  pd.to_datetime(df['Date'], format='%YM%m')
	df = df.set_index('Date')
	return df

def get_money():
	df = ssb.money()[['Tid', 'Data']]
	df = df.rename(columns={'Data':'M2', 'Tid':'Date'})
	df['Date'] =  pd.to_datetime(df['Date'], format='%YM%m')
	df = df.set_index('Date')
	return df

def get_bonds():
	bonds = pd.read_excel(BONDFILE, skiprows=6).dropna()
	bonds = bonds.rename(columns={'Effective date ':'Date', 
							   	  'S&P Norway Sovereign Bond Index':'svr_bond_index_norw'})
	bonds['Date'] = pd.to_datetime(bonds['Date'])
	bonds = bonds.set_index('Date')
	bonds = bonds.resample('ME').last()
	bonds.index = bonds.index.to_period('M').to_timestamp()
	return bonds

def get_titlon():
	con = pymysql.connect(host='titlon.uit.no', 
						user= cr.user, 
						password = cr.password, 
						database='OSE')  
	crsr=con.cursor()
	crsr.execute("SELECT * FROM OSE.equityindex_linked")

	r=crsr.fetchall()
	df=pd.DataFrame(list(r), columns=[i[0] for i in crsr.description])
	df['Date'] = pd.to_datetime(df['Date'])
	df = df.set_index('Date')
	df = df.resample('ME').last()
	df.index = df.index.to_period('M').to_timestamp()
	return df

