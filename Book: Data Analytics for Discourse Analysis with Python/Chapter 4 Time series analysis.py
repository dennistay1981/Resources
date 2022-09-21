#import libraries
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import stats

#generate and plot 100 random values
pd.DataFrame(np.random.normal(0,1,100)).plot()

#compute (P)ACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
series=pd.read_csv('data.csv',index_col='t')
acf(series.y,nlags=10)
pacf(series.y,nlags=10)

#plot (P)ACF
plot_acf(series.y, lags=10, alpha=0.05, title='Autocorrelation function')
plot_pacf(series.y, lags=10, alpha=0.05, title='Partial autocorrelation function') 

#seasonal decomposition of time series
from statsmodels.tsa.seasonal import seasonal_decompose
series.index=pd.date_range(freq='m',start='1949',periods=len(series))
seasonal_decompose(series.y).plot()


"""
STEP 1: Inspect series
"""
#inspect series
series=pd.read_csv('data.csv',index_col='Session')
series.plot(subplots=True)
series.describe()

#ADF test for stationarity
from statsmodels.tsa.stattools import adfuller
adfuller(series.Therapist, autolag='AIC')[1]

#Breusch-Pagan test for homoscedasticity
from statsmodels.formula.api import ols
series['time']=np.array(range(len(series)))
model = ols('Therapist ~ time', data=series).fit()

def create_array(col):
 s = []
 for i in col:
     a = [1,i]
     s.append(a)
 return (np.array(s))
array = create_array(series.Therapist)

from statsmodels.stats.diagnostic import het_breuschpagan
het_breuschpagan(model.resid, array)[1]

#differencing
series['Therapist_diff']= series.Therapist.diff(periods=1)

#log-transformation
series['Therapist_logged']= np.log(series.Therapist)


"""
STEP 2: Compute (P)ACF
"""
#compute (P)ACF
acf(series.Therapist,nlags=10)
pacf(series.Therapist,nlags=10)

#plot (P)ACF
plot_acf(series.Therapist, lags=10, alpha=0.05, title='Autocorrelation function')
plot_pacf(series.Therapist, lags=10, alpha=0.05, title='Partial autocorrelation function') 

#multiple (P)ACF plots
fig,axes=plt.subplots(2,2)
fig.text(0.5, 0.05, 'Therapist vs. Client (analytical thinking)',ha='center',fontsize=15) 
plot_acf(series.Therapist, ax=axes[0,0], alpha=0.05, title='Therapist ACF',lags=10)
plot_pacf(series.Therapist,ax=axes[0,1], alpha=0.05, title='Therapist PACF', lags=10)
plot_acf(series.Client, ax=axes[1,0], alpha=0.05, title='Client ACF',lags=10)
plot_pacf(series.Client,ax=axes[1,1], alpha=0.05, title='Client PACF', lags=10)


"""
STEP 4: Fit model and estimate parameters
"""
#remove final three values to form training data
train_series=series.iloc[0:len(series)-3]

#fit candidate models
import statsmodels.api as sm
model1=sm.tsa.SARIMAX(train_series.Therapist, order=(1,0,0),trend='c').fit()
model1.summary()

model2=sm.tsa.SARIMAX(train_series.Client, order=(0,0,1),trend='c').fit()
model2.summary()

"""
STEP 5: Evaluate predictive accuracy, model fit, and residual diagnostics
"""
#predict and forecast values
predict=(model1.get_prediction(start=1,end=len(series))) 
predictinfo=predict.summary_frame()

forecast=(model1.get_prediction(start=len(series),end=len(series)+3)) 
forecastinfo=forecast.summary_frame()

#plot predicted/forecasted vs. observed values
fig, ax = plt.subplots(figsize=(10, 5))
forecastinfo['mean'].plot(ax=ax, style='k--',label="forecast")
plt.plot(series.Therapist, label="observed",color='dodgerblue')
plt.plot(predictinfo['mean'], label="predicted",color='orange')
ax.axvspan(len(train_series),len(train_series)+3,color='red',alpha=0.2,label='train-test')  
ax.fill_between(forecastinfo.index, forecastinfo['mean_ci_lower'], forecastinfo['mean_ci_upper'], color='grey', alpha=0.2, label="95% CI")
ax.set_ylabel('Analytical thinking', fontsize=12)
ax.set_xlabel('Session', fontsize=12)
ax.set_title('Therapist series',fontsize=12)
plt.setp(ax, xticks=np.arange(1, len(series)+4, step=2))   
plt.legend(loc='best',fontsize=10)

#MAPE for training data (i.e., sessions 1-37)
from sklearn.metrics import mean_absolute_percentage_error
mape_train = mean_absolute_percentage_error(train_series.Therapist, predict.predicted_mean.iloc[0:len(train_series)])    
mape_train*100 

#MAPE for testing data (sessions 38-40)
mape_test = mean_absolute_percentage_error(series.Therapist.iloc[-3:], predict.predicted_mean.iloc[-3:])  
mape_test*100

#R2 for training data 
from sklearn.metrics import r2_score
r2 = r2_score(train_series.Therapist, predict.predicted_mean.iloc[0:len(train_series)])

#obtain residuals from model
residuals = model1.resid

#residual diagnostic plots
fig,axes=plt.subplots(2,2)
fig.tight_layout(pad=2.0)
fig.suptitle('Residual diagnostics',fontsize=14,y=1.05)
plot_acf(residuals, ax=axes[0,0], alpha=0.05, title='ACF of residuals', lags=10)
plot_pacf(residuals, ax=axes[0,1], alpha=0.05, title='PACF of residuals', lags=10)
residuals.plot(ax=axes[1,0], title='Residuals series plot')
sns.distplot(residuals,kde=True, axlabel='Residuals histogram',ax=axes[1,1])

#shapiro-wilk test of normality of residuals
stats.shapiro(residuals)










































































