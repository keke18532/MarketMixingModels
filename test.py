import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tools.tools as sm_tools
import matplotlib.dates as dates
import matplotlib.ticker as tkr
from sklearn.model_selection import train_test_split

df= pd.read_csv('C:\Users\MING\Desktop\liaoke\project\simulated_sales.csv')
y=df['sales']
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape
