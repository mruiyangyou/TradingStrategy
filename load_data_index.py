from jqdatasdk import *
import pandas as pd
import numpy as np
import datetime as dt
from scipy import stats
from statsmodels import regression
from jqdatasdk.technical_analysis import *
import sklearn
import statsmodels.api as sm
import pickle
import math


auth('18261826673','yY204527!')
# measure volatality
def get_atr(codes,dates,timeperiod=14,unit='1d'):
    res=[]
    for date in dates:
        MTR,ATR_=ATR(codes, date , timeperiod=timeperiod, unit = unit,include_now = True)
        tmp=pd.Series(ATR_,name=date)
        res.append(tmp)
    atr=pd.concat(res,axis=1).T
    return atr

#measure macd

def get_macd(codes,dates, SHORT = 12, LONG = 26, MID = 9, unit='1d' ):
    res=[]
    for date in dates:
        dif, dea,macd= MACD(codes, date, SHORT = SHORT, LONG = LONG, MID = MID, unit = unit, include_now = True)
        tmp=pd.Series(macd,name=date)
        res.append(tmp)
    macd=pd.concat(res,axis=1).T
    return macd

# measure rsrs
def get_rsrs(codes, period, window, start, end, unit='1d'):
    res = []
    for code in codes:
        tmp = get_price(code, start_date=start, end_date=end, frequency='daily', fields=None, skip_paused=False,
                        fq='pre')
        #tmp.index = tmp['time']
        beta = pd.Series(index=tmp.index, name=code)
        rsquare = pd.Series(index=tmp.index, name=code)
        high = tmp.high.copy()
        low = tmp.low.copy()

        for i in range(period, len(high)):
            tmpHigh = high.iloc[i - period + 1:i + 1].copy()
            tmpLow = low.iloc[i - period + 1:i + 1].copy()
            if (sum(pd.isnull(tmpHigh)) + sum(pd.isnull(tmpLow))) > 0:
                continue
            x = sm.add_constant(tmpLow)
            model = sm.OLS(tmpHigh, x)
            results = model.fit()
            beta.iloc[i] = results.params.low
            rsquare.iloc[i] = results.rsquared

        mean = beta.rolling(window=window).mean()
        std = beta.rolling(window=window).std()
        beta_std = (beta - mean) / std
        right = beta_std * beta * rsquare

        res.append(right)
    rsrs = pd.concat(res, axis=1)
    return rsrs


# get macro economics indicators
def get_macro():

    q = query(macro.MAC_MANUFACTURING_PMI)
    pmi = macro.run_query(q).set_index('stat_month')[['pmi', 'new_export_orders_idx']].sort_index().pct_change(
        periods=12)

    q = query(macro.MAC_CPI_MONTH).filter(macro.MAC_CPI_MONTH.area_code == 701001)
    cpi = macro.run_query(q).set_index('stat_month')['yoy']

    indexs = list(set(pmi.index) & set(cpi.index))
    indexs.sort()
    df = pd.concat([pmi.loc[indexs], cpi.loc[indexs]], axis=1)
    df.columns = ['pmi', 'export', 'cpi']
    return df


def get_pb_pe(index_code, start, end):
    stocks = get_index_stocks(index_code)
    q = query(valuation).filter(valuation.code.in_(stocks))
    dates = pd.date_range(start, end)
    pb_list = []
    for date in dates:
        df = get_fundamentals(q, date)
        pb_df = df[df['pb_ratio'] > 0].copy()
        pb = pb_df['circulating_market_cap'].sum() / ((pb_df['circulating_market_cap'] / pb_df['pb_ratio']).sum())
        pb_list.append(pb)

    d = {'pb': pd.Series(pb_list, index=dates)}
    pe_pb = pd.DataFrame(d)

    dates, pe_q_list, pb_q_list = [], [], []
    for i in range(252, pe_pb.shape[0]):
        pe_sr = pe_pb.iloc[i - 252:i + 1, 0].copy()
        pb_sr = pe_pb.iloc[:i + 1, 0].copy()
        date = pe_pb.index[i]

        pe_quantile = pe_sr.rank(pct=True, method='max').iloc[-1]
        pb_quantile = pb_sr.rank(pct=True, method='max').iloc[-1]

        dates.append(date)
        pe_q_list.append(pe_quantile)
        pb_q_list.append(pb_quantile)

    d = {'pb_1': pd.Series(pe_q_list, index=dates),
         'pb_his': pd.Series(pb_q_list, index=dates)}

    return pd.DataFrame(d)

# get trade date
def get_tradeDates(start,end,unit='1d'):
    tmp=get_price('000001.XSHE', start_date=start, end_date=end, frequency='daily', fields=None, skip_paused=False, fq='pre')['close']
    return list(pd.to_datetime(tmp.index))

# load data
days=4000
end=dt.datetime.strptime('2020-10-11','%Y-%m-%d')
start= end - dt.timedelta(days=days)
unit='1w'
#codes=['000300.XSHG']
codes = ['000001.XSHE']
features=['ret1','ret2','atr','rsrs','pmi','cpi','export','pb_1','pb_his']

ohlc=get_price(codes, start_date=start, end_date=end, frequency='daily', fields=None, skip_paused=False, fq='pre')
ohlc.set_index(['time'], inplace=True)
ret=ohlc['close'].pct_change(periods=5).dropna()
#ret.columns=['ret']
close=ohlc['close']
close.columns=['close']
#volume=ohlc['volume'].pct_change()
#money=ohlc['money'].pct_change()
atr=get_atr(codes=codes, dates=ret.index,unit=unit)
atr.columns=['atr']
#macd=get_macd(codes=codes, dates=ret.index,unit=unit)
rsrs=get_rsrs(codes,20,60,start,end,unit=unit).dropna()
rsrs.columns=['rsrs']
macro=get_macro().dropna()
pb=get_pb_pe(codes[0],start-dt.timedelta(days=300),end)

#format all the dataframes
for var in ['close','atr','rsrs','pb']:
    eval(var).index=[x.date() for x in eval(var).index]

macro.index=[dt.datetime.strptime(x,"%Y-%m").date() for x in macro.index]
pb.drop(columns='pb_1',inplace=True)
macro.drop(columns='export',inplace=True)

indexs=set(close.index)
for var in ['atr','rsrs','pb']:
    indexs=indexs&set(eval(var).index)
indexs=list(indexs)
indexs.sort()

# merge all the dataframe
data=pd.concat([close.loc[indexs],atr.loc[indexs],rsrs.loc[indexs],pb.loc[indexs]],sort=False,axis=1)
data=data.iloc[::5]
data['ret']=data['close'].pct_change()
data[['atr','rsrs','pb_his']]=data[['atr','rsrs','pb_his']].shift(1)
data['ret1']=data.ret.shift(1)
data['ret2']=data.ret.shift(2)
data.drop(columns=['close'],inplace=True)

dates=macro.index
data=pd.concat([data,pd.DataFrame(columns=['pmi','cpi'],index=data.index)],sort='False',axis=1)
for index,row in data.iterrows():
    date=dates[dates<index][-1]
    data.loc[index,['pmi','cpi']]=macro.loc[date]

data[['pmi','cpi']]=data[['pmi','cpi']].shift(1)

data.to_csv('data_index.csv')



