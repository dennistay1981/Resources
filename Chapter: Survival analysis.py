#Import libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

#Import data file
df=pd.read_csv('data.csv')

#Descriptive plots (Figure 1)
plot1=sns.displot(data=df.loc[(df.Attrition==1)], x="Turns",hue='Target',col='Conventionality',bins=12)
plot1.fig.suptitle('Survival time histograms',y=1.05,fontsize=20)

plot2=sns.catplot(data=df.loc[(df.Attrition==1)],kind='box',y='Turns',x='Conventionality',hue='Target')
plot2.fig.suptitle('Survival time boxplots',y=1.05,fontsize=20)

#Fit Cox’s Proportional Hazards regression model 
coxph = CoxPHFitter()coxph.fit(df, duration_col='Turns', event_col='Attrition', formula="Conventionality*Target")

#Check proportional hazards assumptions 
coxph.check_assumptions(df, p_value_threshold=0.05, show_plots=True)

#Print model summary
coxph.print_summary()
coxph.concordance_index_

#Plot survival curves by both predictors (Figure 2)
ax = plt.subplot(111)

kmf_A = KaplanMeierFitter()
ax = kmf_A.fit(durations=df.loc[(df.Conventionality =='C') & (df.Target =='Org')].Turns, 
               event_observed=df.loc[(df.Conventionality =='C') & (df.Target =='Org')].Attrition, label='C+Org').plot_survival_function(ax=ax)

kmf_B = KaplanMeierFitter()
ax = kmf_B.fit(durations=df.loc[(df.Conventionality =='C') & (df.Target =='Citi')].Turns, 
               event_observed=df.loc[(df.Conventionality =='C') & (df.Target =='Citi')].Attrition, label='C+Citi').plot_survival_function(ax=ax)

kmf_C = KaplanMeierFitter()
ax = kmf_A.fit(durations=df.loc[(df.Conventionality =='N') & (df.Target =='Org')].Turns, 
               event_observed=df.loc[(df.Conventionality =='N') & (df.Target =='Org')].Attrition, label='N+Org').plot_survival_function(ax=ax)

kmf_D = KaplanMeierFitter()
ax = kmf_B.fit(durations=df.loc[(df.Conventionality =='N') & (df.Target =='Citi')].Turns, 
               event_observed=df.loc[(df.Conventionality =='N') & (df.Target =='Citi')].Attrition, label='N+Citi').plot_survival_function(ax=ax)

plt.tight_layout()
plt.title('Survival curve by CONVENTIONALITY and TARGET')
plt.show()















