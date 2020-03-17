#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Below script will requests all submissions between a user defined date range
# The data for each subreddit will be outputted to a separate csv file in the same directory as script

# Imports
import pandas as pd
import requests
import time

# Base url for pushshift api
url = 'https://api.pushshift.io/reddit/search/submission'

# Defining subreddit to pull submission information from
subreddits = ['whiskey', 'scotch', 'bourbon']

# Timer to check for runtime
t0 = time.time()

# Jan 1 2015 00:00:00 GMT = 1420070400
for subreddit in subreddits:
    # Instantiating an empty dataframe to concat into
    df = pd.DataFrame()
    
    # Setting a check date for loop to  begin checking against
    # Jan 1 2020 00:00:00 GMT = 1577836800
    check_date = 1577836800
    
    # earlies_date is used to set initial min_date and to drop submissions outside of date range
    earliest_date = 1420070400
    
    while check_date > 1420070400:
        # Limiting number of calls to be within limits of pushshift API
        time.sleep(.3)
        # Changing the min_date parameter to the current earliest post in dataframe
        params = {
            'subreddit' : subreddit,
            'size'      : 500,
            'before'    : check_date
            }
        # Requesting daata
        req = requests.get(url, params)
        data = req.json()
        # Pull out desired data
        posts = data['data']
        # Appending data to dataframe
        df_temp = pd.DataFrame(posts)
        df = pd.concat([df, df_temp], sort=False)
        # Updating check_date to be earliest date to prevent calling double sets of 
        check_date = df['created_utc'].min()
    

    # Dropping all submissions that were pulled outside of desired timeframe
    df.drop(df.loc[df['created_utc'] < earliest_date].index, inplace=True)

    # Writing to csv
    df.to_csv(f'./{subreddit}_submissions.csv')

# Printing runtime of while loop     
print(time.time() - t0)

