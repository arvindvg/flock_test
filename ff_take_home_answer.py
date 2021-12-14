#!/usr/bin/env python
# coding: utf-8

# # Flock Freight Value Metrics
# 
# ## Goal
# 
# 1. Create some metric(s) which quantifies the value of a carrier relationship.
# 2. Assess the ability to predict this value.
# 3. Summarize the findings. Imagine this needs to be communicated to the carrier relationship team in an understandable way.
# 4. Explain the approach you took in part 1, why you chose it, and what its limitations are.
# 5. Explain what you would ideally do next, if you were doing this in your job and had the time and resources you needed to do it to your satisfaction.
# 
# ## Ideas
# 
# - Customer lifetime value (LTV)
#     - Pros: Standard business metric, units (dollars) are understandable
#     - Cons: Can be hard to model real costs with maketing, call center, etc.  Simple churn model isn't going to work given the sporadic jobs
# - Monthly net revenue
#     - Pros: Can start simple and build, units (dollars per month) are understandable
#     - Cons: not as useful as LTV for planning marketing, acquisition costs, 
# - Churn model using [Convoys](https://better.engineering/convoys/)
#     - Pros: I have been wanting to use the model for a while, it is well-suited to the "rolling start" data, and the trucking name pun is gold
#     - Cons: Learning a new library in limited time, not sure how to leverage for useful predictions
# - Probability of becoming a large revenue customer
#     - Pros: Turns problem into a binary supervised learning problem, demonstrates I can do ML
#     - Cons: Without understanding the domains and goals it is likely to produce garbage
# - Probability of accepting job
#     - Pros: Simple metric, easy to finish in time, likely to be right
#     - Cons: Simple metric, is a less intuitive "value of relationship" measure
#     
# Based on the pros and cons (and limited time) I think I am going to go with monthly net and probability of acceptance.
# 
# ## Future Work
# 
# Top 3 things I would do if this were a real task, in priority order:
# 
# 1. Better understand the business.  Impossible to make a good LTV model without that.
# 2. Learn what value metrics would be useful.  Talk with consumers of the information.
# 3. Spend more time looking at the data.  Get crisp on missing values, seasonality, data generation process. Meaning of some terms needs clarity (see below)
# 
# For the two value metrics I did below there are specific "Future Work" sections embedded below.
# 
# ## Specific future tasks
# 
# - Sometimes `LOADS` is larger than `OFFERS` + `SELF_SERVE_OFFERS` in a single row. Figure out why.
#     - My guess: delay between calendar week of offer and load.  This is supported by the existance of fields like `OFFERS_REVENUE_HAULED`?  If so a data format with a record per offer and proper occurred at timestamps would be better.
#     - Example: Using only 2020 data for ID `9f22382a-efdb-4597-a057-bb5959bcdb00` Loads `768` Offers (self and regular) `195`.
#     - Actually, Loads > Total Offers seems to be true for the majority of carriers. Must be that loads happen without offers?
# - Total Escalations is almost as large as sum of Loads.  That seems very high, so I must be misunderstanding something?
# - Need to understand the mechanism for the partial data in next to last week (see below)
# 
# ## Time Log
# 
# - Friday night: 
#     - ~30 minutes reading documentation, downloading CSV and poking around in Tableau
# - Saturday: 
#     - 9:30-10:45 planning, documentation
#     - 11:15-12:00 math + code for Wilson
#     - 1:00-1:45 code: finished up Wilson + validation
# - Monday
#     - 7:00-8:00 simple net revenue predictions, documentation
#     - 8:30-10:00 pivot to `REVENUE_HAULED`, debugging
#     - 10:00-12:00 regression, more debugging, documentation

# In[ ]:


print(f"Total Hours {(30+75+45+45+60+90+120)/60}")


# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Likelihoood of Delay Escalation
# 
# This was **Probability of Accepting a job**, but something is up with the `Offers` and `Loads` data (see "Specific future tasks" at top).  I had already done the scoring code and didn't want to toss it given the limited time, so I pivoted to likelihood of escalation for a delay.
# 
# ## Goal
# 
# Rank carriers by the likelihood that they will have delay escalations based on their past performance.
# 
# ## Approach
# 
# The obvious approach is to estimate the probability of delay escalation based on the carrier's past performance using something the ratio of delays $D$ to loads $L$
# 
# \begin{equation*}
# p(delay) \approx \frac{D}{L}
# \end{equation*}
# 
# But that can be a bad estimator for small samples.  For example a carrier with a single load that is delayed probably isn't as likely to have delays as a carrier with 100 loads that *all* had delays.
# 
# The solution I like for this is the Wilson score
# 
# \begin{equation*}
# \frac{\alpha + \frac{z^2}{2}}{N+z^2} \pm \frac{z}{N+z^2} \sqrt{\frac{\alpha\left(N-\alpha\right)}{N} + \frac{z^2}{4}}
# \end{equation*}
# 
# For a 0.95 confidence level we can use $z$ = 1.96.
# 
# For background:
# - I really like the article "[how not to sort by average rating](https://www.evanmiller.org/how-not-to-sort-by-average-rating.html)" as an overview for using the Wilson score for problems like this.  Other resources:
# - [This has nice theoretical motivation](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval)
# - Wilson score is also [Reddit's "best" score](https://redditblog.com/2009/10/15/reddits-new-comment-sorting-system/)
# 
# ## Future work
# 
# - Get data with load per row (see notes at top) so that I can actually do percentage of loads with delay rather than this messy weekly sum.
# - Use additional information about the carrier (not just past performance) to make a more predictive model.
# - Explore seasonality (likely to be significant because of weather).

# In[ ]:


df = pd.read_csv("carrier_valuation_takehome_data.csv")
df.head()


# In[ ]:


def wilson(alpha, N, z=1.96, trimAlpha = True):
    if alpha == 0:
        return 0
    
    if alpha > N:
        if trimAlpha:
            alpha = N
        else:
            raise ValueError('Alpha cannot be larger than N')

    z2 = np.square(z)
    wilson1 = np.divide(np.add(alpha, z2/2), 
                        np.add(N, z2))
    wilson2 = np.divide(z, 
                        np.add(N, z2))
    wilson3 = np.divide(np.multiply(alpha, np.subtract(N, alpha)),
                       N)
    wilson4 = np.divide(z2, 4)

    wilsonInterval = np.subtract(wilson1, np.multiply(wilson2, np.sqrt(np.add(wilson3, wilson4))))

    return wilsonInterval

# Let's also calculate the average
def correctedMean(alpha, N):
    return min(alpha,N)/N


# In[ ]:


# used this multiple times, so decided to wrap it as a function
def addScores(rawDf):
    df = rawDf.groupby("ID")['ESCALATIONS_DELAY','LOADS'].sum()
    # Drop carriers with no loads (no prediction)
    df = df[df['LOADS'] != 0]
    # Calculate Delay Wilson Score
    df["DELAY_WILSON_SCORE"] = df.apply(
        lambda df: wilson(df['ESCALATIONS_DELAY'],df['LOADS']), axis=1)
    df["DELAY_MEAN"] = df.apply(
        lambda df: correctedMean(df['ESCALATIONS_DELAY'],df['LOADS']), axis=1)
    return df


# In[ ]:


loadsDf = addScores(df)

loadsDf.head()


# ## Validation 
# 
# One of the goals is "Assess the ability to predict this value."  Let's do a simple chronological validation.
# 
# Validation metric: RMS to start, but for future work this might be a good candidate to use a ranking metric, since that is probably closer to what we care about.
# 
# ### Notes
# 
# - I trried to use only the last week of data, but it was partial and there are only 20 loads, so I went back one more.
# 
# ### Conclusions
# 
# I am pleasantly surprised that the Wilson score has a lower RMSE than the mean, but for this example it did!  I ran it for a few different weeks, and the results are pretty consistent.  Nice!

# In[ ]:


maxDate = "2021-02-15 00:00:00" # "2021-02-22 00:00:00" # max(df['CALENDAR_WEEK'])
validationDf = df[df['CALENDAR_WEEK'] == maxDate]
validationDf = addScores(validationDf)

print(f"{sum(validationDf['LOADS'])} loads")
print(f"{sum(validationDf['ESCALATIONS_DELAY'])} delay escalations")


# In[ ]:


trainDf = df[df['CALENDAR_WEEK'] < maxDate] # I love ISO-8601

trainDf = addScores(trainDf)


# In[ ]:


metricsDf = pd.merge(validationDf, trainDf, on="ID", how="inner", suffixes=("_TEST","_TRAIN"))
metricsDf.head()


# In[ ]:


# Root Mean Square Error
def rmse(x,y):
    return np.sqrt(np.mean((x-y)**2))


# In[ ]:


rmse(metricsDf['DELAY_MEAN_TRAIN'], metricsDf['DELAY_MEAN_TEST'])


# In[ ]:


rmse(metricsDf['DELAY_WILSON_SCORE_TRAIN'], metricsDf['DELAY_MEAN_TEST'])


# # Monthly Net Revenue
# 
# ## Goal
# 
# Predict monthly net revenue per carrier.
# 
# ## Approach
# 
# 1. Figure out which fields relate to revenue and cost (looks like `REVENUE_HAULED` and `ACTUAL_COST`, but I would want to verify that.)
# 2. Do the simple predictions
# 3. <s>Build a simple time series forecast model</s> Ran out of time.  I was just planning a basic regression model, but given the failure of the average to predict I decided it was better to spend the time debugging.
# 
# ### Very simple predictions 
# 
# Let's try a few *very* simple predictions to start:
# 
# 1. Last Week: Predicted week = previous week (per carrier)
# 2. Average: Predicted week = average of all previous weeks (per carrier)
# 3. Zero: since the mode is 0, what happens if I just predict 0 all the time?
# 
# This will establish a baseline for prediction.
# 
# ## Notes
# 
# - I decided to do weekly predictions because of the format of the data.  Just multiply by 30/7 to get monthly.
# - This would normally be a different notebook, but I kept them together to make sharing easier.
# 
# ## Conclusions
# 
# RMSE seems very high compared to average value, meaning these are bad predictions.  The best model of the three is just predicting zero all the time.
# 
# | Model | RMSE |
# | - | - |
# | Last Week | 895 |
# | Average | 398 |
# | Zero | 376 |
# 
# I think we have two problems:
# 
# First, looking at the time series data by carrier it seems like that is because it is pretty sporatic - almost all values are 0.  Predicting rare events is noisy!
# 
# ![image.png](attachment:image.png)
# 
# Second, **net** revenue is much noisier than either gross or costs, because the difference between two numbers that are close greatly magnifies the noise.  I was hoping the averaging would overcome that some, but it clearly didn't (or there is a bug in the code or an error in my understanding of the problem).  Reminds me of [this XKCD comic](https://www.explainxkcd.com/wiki/index.php/2295:_Garbage_Math).
# 
# It is also worth noting that these predictions may be useful for long-term forecasting even if they have high error in single week prediction.
# 
# 
# ## Future Work
# 
# - Do proper LTV by adding sales, marketing, and support costs (see notes at top)
# - Once there is enough history for seasonality, etc. [Facebook Prophet](https://facebook.github.io/prophet/) has given me good reults in the past.
# - ML model with carrier features (didn't attempt because I don't know the problem area / data well enough).

# In[ ]:


# Grab a fresh copy of the data
df = pd.read_csv("carrier_valuation_takehome_data.csv")

# Replace this with a more accurate net value later
df['NET'] = df['REVENUE_HAULED'] - df['ACTUAL_COST']

# Last full week as validation data
maxDate = "2021-02-22 00:00:00" # max(df['CALENDAR_WEEK'])
testSeries = df[df['CALENDAR_WEEK'] == maxDate].set_index("ID")['NET']
trainDf = df[df['CALENDAR_WEEK'] < maxDate]


# In[ ]:


def rmse_join(leftSeries, rightSeries, indexName = "ID"):
    # Pass in two Pandas series with ID index, join on indecies and do rmse
    metricsDf = pd.merge(leftSeries, rightSeries, how="inner", on=indexName)
    return rmse(metricsDf.iloc[:,0],metricsDf.iloc[:,1])


# In[ ]:


## LAST WEEK

# Would have been faster to hard code this, but I wanted to show robust coding
weekAgoDatetime = datetime.fromisoformat(maxDate) - timedelta(days=7)
weekBeforeMaxDate = datetime.isoformat(weekAgoDatetime).replace("T", " ") # ISO mismatch

predSeries = df[df['CALENDAR_WEEK'] == weekBeforeMaxDate].set_index("ID")['NET']
rmse_join(predSeries, testSeries)


# In[ ]:


## AVERAGE
meanNetSeries = trainDf.groupby("ID")['NET'].mean()
rmse_join(meanNetSeries, testSeries)


# In[ ]:


## JUST PREDICT ZERO
rmse_join(testSeries-testSeries, testSeries)


# # Forecasting `REVENUE_HAULED`
# 
# ## Goal
# 
# Predict future values of `REVENUE_HAULED`.
# 
# ## Notes
# 
# - Pros: This should be easier than net revenue because it eliminates the subtraction issue described above.
# - Cons: Possibly not as helpful for planning?
# 
# ### Changes from above work:
# 
# - Looking at `REVENUE_HAULED` in Tableau, it seems like there are is significant temporal correlation ("streaks"), which means that regression models should have predictive power
# 
# ## Conclusions
# 
# Note: RMSE is higher than previous example because of change from net to gross revenue.
# 
# After updating the max date to Feb 15 (see below for motivation) I get:
# 
# | Model | RMSE |
# | - | - |
# | Last Week | 3,014 |
# | Average | 2,402 |
# | Zero | 2,416 |
# 
# Yes!  The average now performs *slightly* better than predicting all zeros.  This gives me hope that this is a tractable problem.

# In[ ]:


# Last full week as validation data (fresh copy)
maxDate = "2021-02-15 00:00:00" # "2021-02-22 00:00:00" # max(df['CALENDAR_WEEK'])

weekAgoDatetime = datetime.fromisoformat(maxDate) - timedelta(days=7)
weekBeforeMaxDate = datetime.isoformat(weekAgoDatetime).replace("T", " ") # ISO mismatch

testSeries = df[df['CALENDAR_WEEK'] == maxDate].set_index("ID")['REVENUE_HAULED']
trainDf = df[df['CALENDAR_WEEK'] < maxDate]


# In[ ]:


## LAST WEEK
predSeries = trainDf[trainDf['CALENDAR_WEEK'] == weekBeforeMaxDate].set_index("ID")['REVENUE_HAULED']
rmse_join(predSeries, testSeries)


# In[ ]:


## AVERAGE
meanSeries = trainDf.groupby("ID")['REVENUE_HAULED'].mean()
rmse_join(meanNetSeries, testSeries)


# In[ ]:


## JUST PREDICT ZERO
rmse_join(meanNetSeries-meanNetSeries, testSeries)


# ## Debugging
# 
# Ok, something seems up here.  I can understand that predicting net revenue is hard, but it seems like using the average to predict `REVENUE_HAULED` should beat zero.
# 
# I explored a bunch of stuff, and when it matched my expectations I deleted it.  Basically looking for where this error could come from.
# 
# In the end I found that the Feb 22 data is likely also partial, which was causing the bad predictions above.  Revising work with the 15th as the validation data.

# In[ ]:


np.mean(meanSeries)


# In[ ]:


# This should be about the same as the mean if data revenue per carrier is even sort of stationary
np.mean(testSeries)


# In[ ]:


# Ergodicity
revenueList = []
dateList = []
maxDate = max(df['CALENDAR_WEEK'])
for weeksAgo in range(16):
    ts = datetime.fromisoformat(maxDate) - timedelta(days=7*weeksAgo)
    date = datetime.isoformat(ts).replace("T", " ") # ISO mismatch
    dateList.append(date)
    revenueList.append(np.sum(df[df['CALENDAR_WEEK'] == date].set_index("ID")['REVENUE_HAULED']))


# In[ ]:


plt.plot(dateList, revenueList, 'o')
plt.xticks(rotation = 90)


# # Regression model
# 
# I could have just stopped where I was, but I wanted to see if a simple linear regression model would beat a simple average.  It also seemed weird to turn this in with no ML at all...

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


# Grab a fresh copy of the data
df = pd.read_csv("carrier_valuation_takehome_data.csv")


# At one point I decided to remove the carriers with zero revenue for all time - there are 26k out of 37k total and I was worried they were causing the model to revert to all zeros!  Common theme with this sparse data.  Prediction for those could be handled independently, or ignored, depending on goals of the metric. 
# 
# It didn't help performance, so I reverted.

# In[ ]:


# # Get rid of entries with all zeros in training data
# revenueSum = df.groupby("ID")['REVENUE_HAULED'].sum()
# nonzeroCarriers = revenueSum[revenueSum!=0].index
# df = df[df['ID'].isin(nonzeroCarriers)]


# In[ ]:


weeksHistory = 6

# Make list of dates so I can assure things are sorted
dateWeeksOffset = 2
dateList = []
for weeksAgo in range(dateWeeksOffset, weeksHistory+dateWeeksOffset+1): # Offset for partial data
    ts = datetime.fromisoformat(max(df['CALENDAR_WEEK'])) - timedelta(days=7*weeksAgo)
    date = datetime.isoformat(ts).replace("T", " ") # ISO mismatch
    dateList.append(date)


# In[ ]:


# Extract and format revenue as a matrix
# I orignially did this with a dictionary and tons of indexing, but it was slow and this is way better!
revenueDf = df.pivot(index='ID', columns='CALENDAR_WEEK',values='REVENUE_HAULED').fillna(0)


# In[ ]:


y = np.array(revenueDf[dateList[0]]) # target
X = np.array(revenueDf[dateList[1:]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1234)

lr = LinearRegression().fit(X_train, y_train)


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


y_pred = lr.predict(X_test)


# In[ ]:


lr.coef_


# In[ ]:


rmse(y_pred, y_test)


# In[ ]:


# Zeros (cross-validation style)
rmse(0, y_test)


# In[ ]:


# Last Week (cross-validation style)
rmse(X_test[:,0], y_test)


# In[ ]:


# Test average prediction in same cross-validation style 
# (omits the ragged start, so calling this "window average")
rmse(np.mean(X_test, 1), y_test)


# ## Regression conclusions
# 
# With six weeks of history:
# 
# | Model | RMSE |
# | - | - |
# | Zeros | 2,252 |
# | Last Week | 3,092 |
# | Window average | 2,379 |
# | Regression | 2,233 |
# 
# Results were pretty similar with 52 weeks history, but vary some depending on the random seed for test/train split.  
# 
# Bottom line: I am not convinced that regression significantly beats window average, and I think they are all beat by the ragged average above.  None are accurate predictors of weekly revenue.

# In[ ]:




