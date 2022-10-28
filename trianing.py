import pandas as pd
from sklearn.naive_bayes import GaussianNB
from metrics import *
import numpy as np

data = pd.read_csv('/Users/marceloyou/Desktop/算法/selecting stock/data_index.csv', index_col = 'Unnamed: 0')
data.index = pd.to_datetime(data.index)
# simple choice
data=data.astype('float')
def classification(x):
    if x<-0.02:
        return -2
    elif -0.02<x<=0:
        return -1
    elif 0<x<0.02:
        return 1
    else:
        return 2

# training
data.dropna(inplace=True)
train_window = 70
x = data[['rsrs', 'cpi', 'atr']].copy()
y = data.ret.apply(classification)
pred, prob, dates, true, ret = [], [], [], [], []
clf = GaussianNB()
for i in range(train_window, data.shape[0]):
    x_train = x.iloc[i - train_window:i].copy()
    y_train = y.iloc[i - train_window:i].copy()
    x_test = np.array(x.iloc[i]).reshape(1, -1)
    y_test = y.iloc[i]

    clf.partial_fit(x_train, y_train, np.unique(y_train))
    pred.append(clf.predict(x_test)[0])
    prob.append(clf.predict_proba(x_test)[0][0])
    dates.append(data.index[i])
    true.append(y_test)
    ret.append(data.ret.iloc[i])
d = {
    'pred': pred,
    'prob': prob,
    'true': true,
    'ret': ret
}

result = pd.DataFrame(d, index=dates)

# Evaluation
# calcualte holdings
position=result.pred.apply(lambda x:0 if x==-2 else 1)
rets_sr=data.ret.loc[position.index]*position
hs300=data.ret.loc[position.index]

# Metrics of the strategy
print('Sharp ratio','-'*10)
print('My strategy:', sharp(rets_sr,5))
print('The index:', sharp(hs300,5))

print('Annual Return', '-'*10)
print('My strategy:',anual_ret(rets_sr,5))
print('The index:',anual_ret(hs300,5))

print('Max Draw:', '-'*10)
print('My strategy:',max_draw(rets_sr))
print('The index:',max_draw(hs300))

print('Prediction Accuracy','-'*10)
true=data.ret.loc[position.index]>0
pred=position
print('My strategy:',win(true,pred))
print('The index:',win(true,pd.Series(1,index=true.index)))

# Return table
netValue(rets_sr,hs300, save = True)

def df_to_markdown(df):
    df = df.reset_index()
    head = pd.DataFrame([df.shape[1]*['---']],index=['---'],columns = df.columns)
    res =head.append(df)
    print('|'+res.to_csv(sep='|',line_terminator='|\n|',index=False)[:-1])

theta=pd.DataFrame(clf.theta_,index=['-2','-1','1','2'],columns=['rsrs','cpi','atr'])
theta=theta.applymap(lambda x:("%.4f")%x)
df_to_markdown(theta)

year_df=pd.DataFrame(index=[str(x) for x in range(2011,2021)],columns=['ret','sharp','draw','win','short'])


rets_sr.index=pd.to_datetime(rets_sr.index)
position.index=pd.to_datetime(position.index)
true=data.ret.loc[position.index]>0
true.index=pd.to_datetime(true.index)

pred=position
pred.index=pd.to_datetime(pred.index)

for index in year_df.index:
    ret_sr=rets_sr[index]
    true_sr=true[index]
    pred_sr=pred[index]
    year_df.loc[index,'ret']=anual_ret(ret_sr,5)
    year_df.loc[index,'sharp']=sharp(ret_sr,5)
    year_df.loc[index,'draw']=max_draw(ret_sr)
    year_df.loc[index,'win']=win(pred_sr,true_sr)
    year_df.loc[index,'short']=sum(position[index]==0)/len(position[index])
year_df=year_df.applymap(lambda x:("%.4f")%x)
df_to_markdown(year_df)

