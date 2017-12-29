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
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

    
df= pd.read_csv('simulated_sales.csv')
y=df['sales']
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
#spliting to train, test
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
    #print models.fit()
    return models
    
        

def model(x_train, x_test, y_train, y_test):
    # Run OLS regression, print summary and return results
    dim_in=30#adjust dim interval
    decay_in=0.3#adjust decay interval
    tv_dim = list(range(120, 151, dim_in))
    tv_decay = list(np.arange(0.6, 0.95, decay_in))
    radio_dim = list(range(150, 181, dim_in))
    radio_decay = list(np.arange(0.3, 0.6, decay_in))
    digital_dim = list(range(70, 101, dim_in))
    digital_decay = list(np.arange(0.6, 0.9, decay_in))
    methods=['s_curve','neg_exp']
    best=[]
    result=[]

    #repeat step 2 for all the combination of parameter to find the best parameter set that have the highest r_squared, or lowest mse, mae.
    for a in tv_dim:
        for b in tv_decay:
            for c in radio_dim:
                for d in radio_decay:
                    for e in digital_dim:
                        for f in digital_decay:
                            for method in methods:
                                kf = KFold(n_splits=2,shuffle=True)
                                iteration=0
                                currentbest=[]
                                final=[]
                                final1=[]
                                #do kfold crossvalidation on train set:
                                for train,val in kf.split(x_train, y_train):
                                    # split the train into k sets,looping over k sets: use 1 set as validating set and k-1 sets as one train set, this is achieved internally in KFold
                                    iteration+=1
                                    print 'split:', iteration
                                    #use train set to build model: model = a_regressor.fit()
                                    print 'train shape:',x_train.iloc[train].shape
                                    train_model=modelfit(method,x_train.iloc[train] ,y_train.iloc[train],a,b,c,d,e,f)
                                    final.append(train_model)
                                    for i in final:
                                        print 'val shape:',x_train.iloc[val].shape
                                        val_model=i.predict(x_train.iloc[val])
                                        mse1=mse(y_train.iloc[val],val_model)
                                        mae1 = mae(y_train.iloc[val],val_model)
                                        arr=[mse1,i]
                                        final1.append(arr)
                                getmax=min(final1, key=lambda x: x[0])
                                for i in final1:
                                    if i[0]==getmax[0]:
                                        currentbest.append(i)
                                        best.append(i)
                                
    for i in best:
        print 'test shape',x_test.shape
        test_model=i[1].predict(x_test)
        print test_model.shape
        mse1 = mse(y_test,test_model)
        mae1 = mae(y_test,test_model)
        print 'mean squared error:',mse1
        print 'mean absolute error:',mae1
        print 'model test summary: ',test_model.summary()
        result.append(i[8])
        return results

result=model(x_train, x_test, y_train, y_test)
plotfi(result, x_test, y_test)
        
    
    
    
