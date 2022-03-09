"""
Functions for various dynamic policy and traditional policy de-identification methods applied to 
simulated subpopulation outbreak detection data.
"""

import numpy as np 
import pandas as pd 




## Dynamic policy - marketer risk approach
# This function prepares the dataset for marketer risk version of WSARE. Though this de-identification
# code and WSARE version were designed for the marketer-based dynamic policy approach, they function to
# simulate any dynamic policy application in which the granularity of the full dataset monotonically
# increases over time.

def marketer_outbreak_detection_deid(df, key, age_name, race_name, sex_name, ethnicity_name):
    
    """
    Function to prepare dataset for WSARE_dynamic_marketer.
    Returns policy indexed dataset and the list of policies to which each policy index value corresponds.
    Policy index value of -1 means do not share.
    """
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # calculate cumulative case counts
    cum_cases = df['date'].value_counts().sort_index().cumsum().fillna(0)
    
    # find minimum value per week
    min_per_week = cum_cases.resample('W-SAT').max().resample('D')\
                            .fillna(method='ffill').shift(periods=1)\
                            .fillna(0).astype(int).to_frame().rename(columns = {'date': 'n_records'})
    
    # create policy index column
    min_per_week['policy'] = -1
    
    # apply de-identification key
    policies = []
    pol_idx = 0
    
    for threshold, policy in key.items():
    
        # add policy parameters to list of policies
        policies.append([age_name[policy[0]], race_name[policy[1]], sex_name[policy[2]], ethnicity_name[policy[3]]])

        # index policy
        min_per_week.loc[min_per_week['n_records'] >= threshold, 'policy'] = pol_idx

        pol_idx += 1
    
    # add policy index to original data
    policy_indexed_df = pd.merge(df, min_per_week.drop('n_records', axis=1), left_on='date', right_index=True)

    return policy_indexed_df, policies




## Dynamic policy - PK risk approach
# This function considers the case where demographic granularity fluctuates with the number of new case records
# over time, without changing the generalization of previously released records.

def PK_outbreak_detection_deid(df, lag, key, age_name, race_name, sex_name, ethnicity_name):
    
    """
    Function to prepare dataset for WSARE_dynamic_PK.
    Returns policy indexed dataset and the list of policies to which each policy index value corresponds.
    Policy index value of -1 means do not share.
    """

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # calculate rolling sum of cases counts
    cases = df['date'].value_counts().sort_index().resample('D').sum().fillna(0)\
                        .rolling(lag, closed='right').sum().fillna(0)
    
    # calculate minimum case count per week (that is not zero)
    min_per_week = cases[cases.gt(0)].resample('W-SAT').min().resample('D').fillna(method='bfill')\
                        .to_frame().rename(columns={'date':'n_records'})
    
    # define policy index column, default is -1 (don't share)
    min_per_week['policy'] = -1
    
    # apply de-identification key
    policies = []
    pol_idx = 0
    
    for threshold, policy in key.items():
    
        # add policy parameters to list of policies
        policies.append([age_name[policy[0]], race_name[policy[1]], sex_name[policy[2]], ethnicity_name[policy[3]]])

        # index policy
        min_per_week.loc[min_per_week['n_records'] >= threshold, 'policy'] = pol_idx
        
        pol_idx += 1
    
    # add policy index to original data
    policy_indexed_df = pd.merge(df, min_per_week.drop('n_records', axis=1), left_on='date', right_index=True, how='left').fillna(method='bfill')

    return policy_indexed_df.astype({'policy':int}), policies




## CDC Public Access with Geography Policy

def CDCpubwGEO_outbreak_detection_deid(df):
    
    """
    Function to de-identify the dataset following the policy applied to the CDC's Public Access with Geography
    dataset. Returns dataset with 'days' column to be used as date column in WSARE module. This is because WSARE
    assumes the date column is in days. Also returns a date_key, which can be used to convert the 'days' column back
    to months. The date_key additionally includes a month_detected column, which is one month after diagnosis date.
    This is because it is assumed all of the cases for a given month are not released until the first day of the following
    month - a caveat of retrospective de-identification.
    """
    
    # copy dataset
    df = df.copy()
    
    # add age_group
    df['age_group'] = pd.cut(df['age'], [0, 18, 50, 65, 150], right=False)
    
    # generalize race
    df['new_race'] = df['race']
    df.loc[(df.race == 'mixed')|(df.race == 'other'), 'new_race'] = 'multiple/other'
    
    df.drop('race', axis=1, inplace=True)
    df.rename(columns = {'new_race': 'race'}, inplace=True)
    
    # Note: ethnicity and sex are not generalized
    
    # add case_month
    df['case_month'] = pd.to_datetime(df['date'].dt.to_period('M').astype(str))
    
    # create date_key
    uniq_months = df.case_month.unique()
    uniq_months.sort()

    date_key = pd.DataFrame({'months' : uniq_months})
    date_key['days'] = pd.date_range('2020-01-01', periods=len(date_key)).tolist() # arbitrary start date

    # add date detected, which is the beginning of the month following case_month
    date_key['month_detected'] = date_key['months'] + pd.DateOffset(months=1)

    # add new 'days' column to dataset
    df_newdate = pd.merge(df, date_key, left_on = 'case_month', right_on = 'months')\
                   .drop(['date', 'months', 'month_detected'], axis=1)
    
    return df_newdate, date_key




## Counts by feature value policy

def Groupby_counts_outbreak_detection_deid(df, ft_cols, age_group = True):
    
    """
    Transforms the dataset into a form that mimics the the information provided by groupby 
    counts by feature value when processed by WSARE algorithm. This module effectively creates
    a new record for each unique value of each demographic quasi-identifier, where there is only one
    feature column. Returns transformed dataset.
    """
    
    # copy dataset
    df = df.copy()

    # create age groups
    if age_group:
        df['age_group'] = pd.cut(df['age'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 120], right=False)

    # for counting records
    df = df.reset_index().astype(str)
    
    # create new records by feature
    records = []

    for ft in ft_cols:

        groupby_ft = df.groupby([ft] + ['date']).agg({'index':'count'}).reset_index()
        groupby_ft = groupby_ft[groupby_ft['index'] > 0].reset_index(drop=True)

        records_ft = np.concatenate(list(map(lambda row: np.tile(groupby_ft.loc[row, [ft] + ['date']],
                                                                 [groupby_ft.iloc[row]['index'], 1]),
                                             range(len(groupby_ft)))))

        records.append(records_ft)
    
    # format results
    new_df = pd.DataFrame(data = np.concatenate(records, axis=0),
             columns = ['feature', 'date'])
    new_df['date'] = pd.to_datetime(new_df['date'])
    
    return new_df.sort_values('date').reset_index(drop = True)
    





