import numpy as np
import pandas as pd
import ssb
import credentials as cr
import pymysql
import paneltime as pt
import os
from statsmodels import api as sm
from matplotlib import pyplot as plt
from get_data_ssb import get_data


# import data from titlon
#	* Stock price index to predict inflation
#	* Bonds, as reference point

# Use inflation data from ssb
# Use paneltime to predict

FLDR = os.path.dirname(__file__)
PREDFILE = f'{FLDR}/output/pred_nor.dmp'
FIGFILE = f'{FLDR}/figures/pred_nor.png'


def main():
	dfd = get_data()

	pt.options.pqdkm = (0,0,0,0,0)

	if os.path.exists(PREDFILE) and False:
		pred = pd.read_pickle(PREDFILE)
	else:
		pred = prediction(dfd)

	fig, ax = plt.subplots()
	fig = ax.scatter(pred['GDP change'], pred['GDP change pred']).get_figure()
	x = np.linspace(pred['GDP change'].min(), pred['GDP change'].max(), 100)
	
	m = OLS(pred, pred['GDP change'], pred['GDP change pred'])
	print(m.summary())
	ax.plot(x , x, color='lightgray', label='45 degree line')
	ax.plot(x, m.predict(sm.add_constant(x)), color='red', label='OLS fit')
	ax.set_xlabel('Actual GDP change')
	ax.set_ylabel('Predicted GDP change')
	ax.set_title('GDP Prediction')
	ax.legend()

	fig.savefig(FIGFILE)
	a=0

def prediction(dfd):
	pred = pd.DataFrame(columns=['Date', 'BNP', 'BNP pred' ])
	for i in range(50, len(dfd)):
		df = dfd.iloc[:i]
		bnp_pred = estimate(df)
		df_next = pd.DataFrame({'GDP change': [ dfd.iloc[i]['BNP']],'GDP change pred':  [bnp_pred]}, index= [dfd.index[i]])
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
	m = pt.execute('BNP ~  L(svr_bond_index_norw,2)+L(OSEBXLinked,1)+L(M2,1) + L(KPI,1)+Q_2+Q_3+Q_4'
		, df, 'Date')
	#m = pt.execute('KPI ~ L(BNP,1)'
	#	, df, 'Date')
	pr = m.predict()
	return pr.iloc[-1]['Predicted BNP']



main()