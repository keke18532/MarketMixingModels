import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tools.tools as sm_tools
import matplotlib.dates as dates
import matplotlib.ticker as tkr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score

df= pd.read_csv('simulated_sales.csv')
y=df['sales']
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print x_train.shape, y_train.shape
print x_test.shape, y_test.shape

def s_curve(grps, dim, decay):
    #s_curve adstock model
    ads = range(len(grps))#adstock
    ads[0] = grps.iloc[0] 
    for i in range(1,len(grps)):
        #print ads[i]
        ads[i] = 1/(1+np.exp(-dim*grps.iloc[i]))+decay*ads[i-1]
    return ads

def neg_exp(grps, dim, decay):
    #negative exponential decay model
    ads = list(range(len(grps)))#adstock
    ads[0] = grps.iloc[0]
    for i in range(1,len(grps)): 
        ads[i] = 1-np.exp(-dim*grps.iloc[i])+decay*ads[i-1]
    return ads

def model(x_train, x_test, y_train, y_test):
    # Run OLS regression, print summary and return results
    tv_dim = list(range(120, 151, 30))
    tv_decay = list(np.arange(0.6, 0.95, 0.34))
    radio_dim = list(range(150, 181, 30))
    radio_decay = list(np.arange(0.3, 0.6, 0.2))
    digital_dim = list(range(70, 101, 30))
    digital_decay = list(np.arange(0.6, 0.9, 0.2))
    final=[]
    mini=[]
    result=[]
    #method 1:
    for a in tv_dim:
        for b in tv_decay:
            for c in radio_dim:
                for d in radio_decay:
                    for e in digital_dim:
                        for f in digital_decay:
                            print 'a b c d e f:',a,b,c,d,e,f
                            tv_train_ads=s_curve(x_train['tv_grps'],a,b)
                            radio_train_ads=s_curve(x_train['radio_grps'],c,d)
                            digital_train_ads=s_curve(x_train['digital_grps'],e,f)
                            sales_train=y_train
                            temp_train=x_train['temp']
                            x_train_ad=pd.concat([x_train['tv_grps'],pd.Series(tv_train_ads),x_train['radio_grps'],pd.Series(radio_train_ads),x_train['digital_grps'],pd.Series(digital_train_ads),temp_train,pd.Series(sales_train)])
                            modelfit=sm.ols(formula='sales_train ~ tv_train_ads+radio_train_ads+digital_train_ads+temp_train',data=x_train_ad).fit()
                            arr=[modelfit.rsquared,a,b,c,d,e,f]
                            final.append(arr)
    getmin=min(a for (a,b,c,d,e,f,g) in final)
    print 'train minimum r squared value:',getmin
    for i in final:
        if i[0]==getmin:
            mini.append(i)
    for i in mini:
        print 'tv_dim, tv_decay, radio_dim, radio_decay, digital_dim, digital_decay: ', i[1], i[2], i[3], i[4], i[5], i[6], '\n'
        tv_test_ads=s_curve(x_test['tv_grps'],i[1],i[2])
        radio_test_ads=s_curve(x_test['radio_grps'],i[3],i[4])
        digital_test_ads=s_curve(x_test['digital_grps'],i[5],i[6])
        sales_test=y_test
        temp_test=x_test['temp']
        x_test_ad=pd.concat([x_test['tv_grps'],pd.Series(tv_test_ads),x_test['radio_grps'],pd.Series(radio_test_ads),x_test['digital_grps'],pd.Series(digital_test_ads),temp_test,pd.Series(sales_test)])
        modeltest=sm.ols(formula='sales_test ~ tv_test_ads+radio_test_ads+digital_test_ads+temp_test',data=x_test_ad).fit()
        print 'model test r squared: ',modeltest.rsquared
        result.append(modeltest)
    return result

result=model(x_train, x_test, y_train, y_test)


