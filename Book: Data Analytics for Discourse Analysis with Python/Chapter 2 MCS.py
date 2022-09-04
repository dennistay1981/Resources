"""
Chapter 2 Monte Carlo Simulations
"""
#import Python libraries
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import scipy as sp
import seaborn as sns



"""
Birthday paradox (analytic solution)
"""
#a function to calculate the probability p given n number of people
def probsamebirthday(n):
    q = 1
    for i in range(1, n):
        probability = i / 365
        q *= (1 - probability)
    p = 1 - q
    return(p)

#runs the function for n=23. feel free to change the value of n.
probsamebirthday(23)




"""
Birthday paradox (numerical solution)
"""
#fix the random seed to ensure reproducibility
np.random.seed(0)

#a function to check for duplicates in a list of birthdays
def contains_duplicates(X):
    return len(np.unique(X)) != len(X)  


#specify no. of people and trials. feel free to change the values.
n_people=23
n_trials=5000


#a for-loop to generate birthdays and calculate the probability for the specified no. of people and trials
list=[]

for i in range (0,n_trials):
    #loop through the number of trials. For each, draw random birthdays for n_people
    dates = np.random.choice(range(1,366), size=(n_people))
    #use the function above to check for duplicates in each trial, appending the result to list
    list.append(contains_duplicates(dates))  

#calculate the final probability as the fraction of all trials where there are duplicates
probability=len([p for p in list if p == True]) / len(list)
print("With", n_trials, "trials, probability of at least one shared bday among",n_people,"people =", probability)



#a more complex for-loop to track and plot the probability for an increasing number of trials from 0 to n_trials
n_people=23
n_trials=5000
trial_count = 0
shared_birthdays = 0
probabilities = []


for experiment_index in range(0,n_trials):
  dates = np.random.choice(range(1,366), size=(n_people))
  if contains_duplicates(dates):
    shared_birthdays+=1 
  trial_count += 1 
  probability = shared_birthdays/trial_count
  probabilities.append(probability)

#plot the outcome (probability) for an increasing number of trials
plt.plot(probabilities)
plt.title("Outcome of " +str(n_trials)+" trials converging to p=" + str(probabilities[-1]))
plt.show()



"""
Casino roulette simulation
"""
#fix the random seed to ensure reproducibility
np.random.seed(0)

#a function to calculate winnings per spin
def winnings(x):
    if x<=55:
        return -1
    else:
        return 1


#specify no. of spins per set and no. of sets. feel free to change the values.
spins=100
sets=1000


#a for-loop to spin the wheel, calculate and keep track of winnings 
list=[]

for i in range(0,sets):
    #loop through the number of sets and spin the specified number of times
    x=np.random.uniform(1,100,spins)   
    #keep track of spin outcomes in a dataframe
    df=pd.DataFrame(data={'x':x})
    #use the function above to determine amount won per spin
    df['Winning']=df['x'].apply(winnings)
    #sum the amount won for all spins, to obtain and record the total winnings per set
    list.append([df['Winning'].sum()])

#plot the distribution of winnings over all the sets
ax=sns.distplot(list,kde=False)
ax.set(xlabel='Winnings', ylabel = 'No. of sets ('+str(spins)+' spins each)')
ax.axvline(x=np.mean(list), color='blue', label='Mean', linestyle='dotted', linewidth=2)
ax.legend()
plt.title('Mean winnings=' +str(np.mean(list)) + '\n St dev winnings=' +str(np.round(np.std(list),decimals=3)))



"""
MCS
"""

#Generate random training set
data=pd.read_csv('40_sessions.csv',index_col='Session')
from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.25,random_state=None)
train.to_csv('train_random.csv')
test.to_csv('validation_random.csv')


#load data
data=pd.read_csv('train_A.csv',index_col='Session')

#calculate parameters for each variable 
ana_avg = np.mean(data.Analytic)
ana_sd = np.std(data.Analytic)
auth_avg = np.mean(data.Authentic)
auth_sd = np.std(data.Authentic)
clout_avg = np.mean(data.Clout)
clout_sd = np.std(data.Clout)
tone_avg = np.mean(data.Tone)
tone_sd = np.std(data.Tone)

#specify no. of sessions and simulations. feel free to change the values.
num_sessions = 40
num_simulations = 5000

"""
Option 1: Simulation without stratified sampling for variance reduction. More straightforward
"""

#a for-loop to draw random variable scores for sessions, calculate and keep track of the mean scores
np.random.seed(0)
allstats=[]
for i in range (num_simulations):
    
    #loop through the number of simulations, drawing random values for each variable
    analytic= np.random.normal(ana_avg, ana_sd, num_sessions)
    authentic= np.random.normal(auth_avg, auth_sd, num_sessions)
    clout= np.random.normal(clout_avg, clout_sd, num_sessions)
    tone= np.random.normal(tone_avg, tone_sd, num_sessions)
    #keep track of simulated scores in a dataframe
    df=pd.DataFrame(index=range(num_sessions),data={'Analytic':analytic, 'Authentic':authentic, 'Clout':clout, 'Tone':tone}) 
    #calculate and store the mean scores 
    allstats.append([df['Analytic'].mean(),df['Authentic'].mean(),df['Clout'].mean(),df['Tone'].mean()])
   
    
"""
Option 2: Simulation with stratified sampling for variance reduction. More complicated
"""
#specify number of strata
num_strata = 10
    
#a for-loop to draw random variable scores for sessions, calculate and keep track of the mean scores
np.random.seed(0)
allstats=[]
for i in range (num_simulations):

    #distribute num_sessions evenly along num_strata
    L=int(num_sessions/num_strata)
    #allocate the probability space 0-1 evenly among the strata
    lower_limits=np.arange(0,num_strata)/num_strata
    upper_limits=np.arange(1, num_strata+1)/num_strata
    #generate random numbers that are confined to the allocated probability space within each stratum. each random number represents the cumulative distribution function for a normal distribution
    points=np.random.uniform(lower_limits, upper_limits, size=[int(L),num_strata]).T
    #create a vector of z-scores, each corresponding to the CDF values calculated above
    normal_vector=sp.stats.norm.ppf(points)
    
    #For each of the four summary variables, generate a vector of normally distributed scores (one score per session) using the normal vector above
    analytic_vector=ana_avg+(ana_sd*(normal_vector))
    analytic_strata_mean=np.mean(analytic_vector, axis=1)
    analytic=np.mean(analytic_strata_mean)
   
    authentic_vector=auth_avg+(auth_sd*(normal_vector))
    authentic_strata_mean=np.mean(authentic_vector, axis=1)
    authentic=np.mean(authentic_strata_mean)
    
    clout_vector=clout_avg+(clout_sd*(normal_vector))
    clout_strata_mean=np.mean(clout_vector, axis=1)
    clout=np.mean(clout_strata_mean)
    
    tone_vector=tone_avg+(tone_sd*(normal_vector))
    tone_strata_mean=np.mean(tone_vector, axis=1)
    tone=np.mean(tone_strata_mean)
    #keep track of simulated scores in a dataframe
    df=pd.DataFrame(index=range(num_sessions),data={'Analytic':analytic, 'Authentic':authentic, 'Clout':clout, 'Tone':tone}) 
    #calculate and store the mean scores 
    allstats.append([df['Analytic'].mean(),df['Authentic'].mean(),df['Clout'].mean(),df['Tone'].mean()])


#convert to dataframe and summarize final outcomes
results_df=pd.DataFrame.from_records(allstats,columns=['Analytic','Authentic','Clout','Tone']) 
results_df.describe().round(3)
    

#plot histograms of final outcomes
fig,axes=plt.subplots(2,2)
fig.suptitle('Simulation (B)')
sns.distplot(results_df.Analytic,kde=False,ax=axes[0,0],axlabel='Analytical thinking')
sns.distplot(results_df.Authentic,kde=False,ax=axes[0,1],axlabel='Authenticity')
sns.distplot(results_df.Clout,kde=False,ax=axes[1,0],axlabel='Clout')
sns.distplot(results_df.Tone,kde=False,ax=axes[1,1],axlabel='Emotional tone')
axes[0,0].text(0.5,0.5,f'M={np.round(np.mean(results_df.Analytic),3)},SD={np.round(np.std(results_df.Analytic),3)}', 
               ha="center", va="top",transform=axes[0,0].transAxes) 
axes[0,1].text(0.5,0.5,f'M={np.round(np.mean(results_df.Authentic),3)},SD={np.round(np.std(results_df.Authentic),3)}', 
               ha="center", va="top",transform=axes[0,1].transAxes) 
axes[1,0].text(0.5,0.5,f'M={np.round(np.mean(results_df.Clout),3)},SD={np.round(np.std(results_df.Clout),3)}', 
               ha="center", va="top",transform=axes[1,0].transAxes) 
axes[1,1].text(0.5,0.5,f'M={np.round(np.mean(results_df.Tone),3)},SD={np.round(np.std(results_df.Tone),3)}', 
               ha="center", va="top",transform=axes[1,1].transAxes)



#perform Welch's t-test for validation
valid=pd.read_csv('validation_A.csv',index_col='Session')
sp.stats.ttest_ind(results_df.Analytic, valid.Analytic, equal_var=False) #Analytic
sp.stats.ttest_ind(results_df.Authentic, valid.Authentic, equal_var=False) #Authentic
sp.stats.ttest_ind(results_df.Clout, valid.Clout, equal_var=False) #Clout
sp.stats.ttest_ind(results_df.Tone, valid.Tone, equal_var=False) #Tone

#Join the two dataframes (results_df and valid) to prepare barplots
joint = pd.concat([results_df, valid], axis=0, ignore_index=False)
joint['Dataset'] = (len(results_df)*('Simulated set',) + len(valid)*('Validation set',))

#plot simulated and validation set variable means
joint.groupby('Dataset').mean().plot(kind='bar',title='Validation (A)').legend(loc='best')
plt.xticks(fontsize=10,rotation=0)
plt.yticks(fontsize=10,rotation=0)
plt.legend(fontsize=10)



