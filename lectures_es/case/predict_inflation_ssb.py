import numpy as np
import pandas as pd
import ssb
import credentials as cr
import pymysql
import paneltime as pt
import os
from statsmodels import api as sm


# import data from titlon
#	* Stock price index to predict inflation
#	* Bonds, as reference point

# Use inflation data from ssb
# Use paneltime to predict

FLDR = os.path.dirname(__file__)
DATAFILE = f'{FLDR}/output/pred_nor_data.dmp'
PREDFILE = f'{FLDR}/output/pred_nor.dmp'
FIGFILE = f'{FLDR}/output/pred_nor.png'
BONDFILE = f'{FLDR}/data/spbonds_nor.xls'

def main():
	df = get_data()

	dfd = np.log(df)
	dfd = dfd.diff()
	
	quarter = np.array((dfd.index.month-1)/3, dtype=int)+1
	
	df_dum = pd.get_dummies(quarter, 
						 drop_first = True, 
						 prefix = 'Q').set_index(dfd.index)
	dfd = pd.concat((dfd,df_dum), axis = 1)

	pt.options.pqdkm = (0,0,0,0,0)

	dfd['time'] = (dfd.index.year-dfd.index.year[0])*12 + dfd.index.month

	if os.path.exists(PREDFILE) and False:
		pred = pd.read_pickle(PREDFILE)
	else:
		pred = prediction(dfd)

	fig = pred.plot.scatter(x='KPI', y='KPI pred').get_figure()
	fig.savefig(FIGFILE)
	m = OLS(pred, pred['KPI'], pred['KPI pred'])
	print(m.summary())
	fig.show()
	a=0

def prediction(dfd):
	pred = pd.DataFrame(columns=['Date', 'KPI', 'KPI pred' ])
	for i in range(50, len(dfd)):
		df = dfd.iloc[:i]
		kpi_pred = estimate(df)
		df_next = pd.DataFrame({'KPI': [ dfd.iloc[i]['KPI']],'KPI pred':  [kpi_pred]}, index= [dfd.index[i]])
		pred = pd.concat((pred, df_next), axis=0)
		a=0

	pred.to_pickle(PREDFILE)
	return pred

def OLS(df, y, x):
	""" Ordinary least squares regression
	"""
	x = sm.add_constant(x)
	model = sm.OLS(y, x).fit()
	return model

def estimate(df):
	m = pt.execute('KPI ~  L(svr_bond_index_norw,2)+L(OSEBXLinked,1)+L(M2,1) + L(BNP,1)+Q_2+Q_3+Q_4'
		, df, 'Date')
	m = pt.execute('KPI ~ L(BNP,1)'
		, df, 'Date')
	pr = m.predict()
	return pr.iloc[-1]['Predicted KPI']

def get_data():
	if os.path.exists(DATAFILE):
		data = pd.read_pickle(DATAFILE)
		return data
	bnp = get_bnp()
	kpi = get_kpi()
	bonds = get_bonds()
	indx = get_titlon()
	money = get_money()
	data = pd.concat((kpi, bonds, indx, bnp, money), axis=1).dropna()
	data = data[['KPI',  'svr_bond_index_norw',  'OSEBXLinked',   'BNP', 'M2']]
	data.to_pickle(DATAFILE)
	
	return data





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

main()