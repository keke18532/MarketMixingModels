import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tools.tools as sm_tools
import matplotlib.dates as dates
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


    
df= pd.read_csv('simulated_sales.csv')
y=df['sales']
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print x_train.shape, y_train.shape
print x_test.shape, y_test.shape

def adsmodel(method, grps, dim, decay):
    if method=='s_curve':
        ads = range(len(grps))#adstock
        ads[0] = grps.iloc[0] 
        for i in range(1,len(grps)):
            ads[i] = 1/(1+np.exp(-dim*grps.iloc[i]))+decay*ads[i-1]
        return ads
    if method=='neg_exp':
        #negative exponential decay model
        ads = list(range(len(grps)))#adstock
        ads[0] = grps.iloc[0]
        for i in range(1,len(grps)): 
            ads[i] = 1-np.exp(-dim*grps.iloc[i])+decay*ads[i-1]
        return ads
    else:
        print 'Model not available.'

def plotfi(result, x_test, y_test):
    a=x_test['week'].argsort()
    model=result[0]#pick the first model
    f, axarr = plt.subplots(2, sharex=True)
    f2, axarr2 = plt.subplots(2, sharex=True)
    axarr[0].set_title('Sales')
    axarr[1].set_title('Grps')
    axarr2[0].set_title('Sales')
    axarr2[1].set_title('Temperature')
    axarr[1].set_xlabel('Week')
    axarr[0].set_ylabel('Sales')
    axarr[1].set_ylabel('Grps')
    axarr2[1].set_xlabel('Week')
    axarr2[0].set_ylabel('Sales')
    axarr2[1].set_ylabel('Temp')
    axarr[0].plot(x_test['week'].iloc[a], y_test.iloc[a], 'o-', label="Predicted Sales")
    axarr[0].plot(x_test['week'].iloc[a],model.fittedvalues.iloc[a], 'ro-', label="Predicted sales using model")
    axarr[0].legend(['Sales', 'Predicted sales using model'])
    axarr[1].bar(x_test['week'].iloc[a], x_test['tv_grps'].iloc[a], color='b', align='center', label='tv_grps')
    axarr[1].bar(x_test['week'].iloc[a], x_test['radio_grps'].iloc[a], color='g', align='center', label='radio_grps')
    axarr[1].bar(x_test['week'].iloc[a], x_test['digital_grps'].iloc[a], color='r', align='center', label='digital_grps')
    axarr[1].legend(['tv_grps', 'radio_grps','digital_grps'])
    axarr2[0].plot(x_test['week'].iloc[a], y_test.iloc[a], 'o-', label="Predicted Sales")
    axarr2[0].plot(x_test['week'].iloc[a],model.fittedvalues.iloc[a], 'ro-', label="Predicted sales using model")
    axarr2[0].legend(['Sales', 'Predicted sales using model'])
    axarr2[1].plot(x_test['week'].iloc[a], x_test['temp'].iloc[a], 'o-', label="temp")
    axarr2[1].legend(['temp'])
    plt.show()


def modelfit(method,x_data,y_data,a,b,c,d,e,f):
    tv_ads=adsmodel(method,x_data['tv_grps'],a,b)
    radio_ads=adsmodel(method,x_data['radio_grps'],c,d)
    digital_ads=adsmodel(method,x_data['digital_grps'],e,f)
    sales=y_data
    temp=x_data['temp']
    week=x_data['week']
    x_ad=pd.concat([x_data['tv_grps'],pd.Series(tv_ads),x_data['radio_grps'],pd.Series(radio_ads),x_data['digital_grps'],pd.Series(digital_ads),temp,week,pd.Series(sales)])
    models=sm.ols(formula='sales ~ tv_ads+radio_ads+digital_ads+temp+week',data=x_ad).fit()
    return models
    
        

def model(method, x_train, x_test, y_train, y_test):
    # Run OLS regression, print summary and return results
    print 'adsmodel name:',method
    dim_in=30#adjust dim interval
    decay_in=0.3#adjust decay interval
    tv_dim = list(range(120, 151, dim_in))
    tv_decay = list(np.arange(0.6, 0.95, decay_in))
    radio_dim = list(range(150, 181, dim_in))
    radio_decay = list(np.arange(0.3, 0.6, decay_in))
    digital_dim = list(range(70, 101, dim_in))
    digital_decay = list(np.arange(0.6, 0.9, decay_in))
    best=[]
    final=[]
    result=[]
    iteration=0
    kf = KFold(n_splits=2,shuffle=True)
    for train,val in kf.split(x_train, y_train):
        currentbest=[]
        iteration+=1
        print 'split:', iteration
        #method 1:
        for a in tv_dim:
            for b in tv_decay:
                for c in radio_dim:
                    for d in radio_decay:
                        for e in digital_dim:
                            for f in digital_decay:
                                #print 'a b c d e f:',a,b,c,d,e,f
                                #np.array(x_train)[train]
                                #print pd.DataFrame(np.array(x_train)[train])
                                train_model=modelfit(method,x_train.reindex(train) ,y_train.reindex(train),a,b,c,d,e,f)
                                arr=[train_model.rsquared,a,b,c,d,e,f]
                                final.append(arr)
                                #best.append(arr)
                                                                    
        getmax=max(a for (a,b,c,d,e,f,g) in final)
        print 'xtrain_train maximum r squared value:',getmax
        for i in final:
            if i[0]==getmax:
                currentbest.append(i)
                best.append(i)
        for i in currentbest:
            #print 'tv_dim, tv_decay, radio_dim, radio_decay, digital_dim, digital_decay: ', i[1], i[2], i[3], i[4], i[5], i[6], '\n'
            val_model=modelfit(method,x_train.reindex(val) ,y_train.reindex(val),i[1], i[2], i[3], i[4], i[5], i[6])
    for i in best:
        print 'tv_dim, tv_decay, radio_dim, radio_decay, digital_dim, digital_decay: ', i[1], i[2], i[3], i[4], i[5], i[6], '\n'
        test_model=modelfit(method,x_test ,y_test,i[1], i[2], i[3], i[4], i[5], i[6])
        #print test_model.fittedvalues.shape
        print 'model test r squared: ',test_model.rsquared
        mse = np.mean((y_test - test_model.fittedvalues)**2)#mean squared error
        mae = np.sum(np.absolute(y_test - test_model.fittedvalues))
        print 'mean squared error:',mse
        print 'mean absolute error:',mae
        print 'model test summary: ',test_model.summary()
        result.append(test_model)
        return result

result=model('neg_exp',x_train, x_test, y_train, y_test)
plotfi(result, x_test, y_test)
        
    
    
    
