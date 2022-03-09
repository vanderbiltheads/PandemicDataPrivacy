"""
Complete set of WSARE variations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from functools import partial
from multiprocessing import Pool
import itertools


## WSARE classes

# Includes at least one class for vanilla WSARE, dynamic PK policy WSARE, and dynamic marketer WSARE.
# Each class has methods for single component and n-component searches, as well as non-parallel and parallel applications.

class WSARE:
    
    """
    Vanilla WSARE. Does not consider varying generalization/transformation of feature values.
    """
    def __init__(self, df, feature_columns, date_column,
                 baseline_days = [56, 49, 42, 35],
                 alpha_single = 0.05, alpha_n = 0.05,
                 randomization = None, n_randomizations = 100):
        
        self.df = df.copy().sort_values(date_column).reset_index(drop=True)
        self.ft_col = feature_columns
        self.dt_col = date_column
        self.alpha_single = alpha_single
        self.alpha_n = alpha_n
        self.alpha = alpha_single
        self.date_diff = np.array(baseline_days)
        self.randomization = randomization
        self.n_rands = n_randomizations
        
        if randomization is not None:
            self.rng = np.random.default_rng()
            if n_randomizations < 100: 
                raise ValueError("Increase number of randomizations to >= 100 or set randomization = False")
        
    def get_unique_feature_values(self):
        
        """
        Stores a list of the unique set of values for each feature in feature_columns.
        """
        
        self.ft_dict={}
        for ft in self.ft_col:
            
            uniq_vals = self.df[ft].unique()
            
            # if there is only one unique value for a feature column,
            # remove the column from consideration
            if len(uniq_vals) == 1:
                
                self.remaining_ftcols.remove(ft)
                
            else:
                
                self.ft_dict[ft] = uniq_vals
            
    def index_dates(self):
        
        """
        Indexes the dataset by date. Considers dates not included in dataset.
        """
        
        # define numeric date index
        self.date_index = pd.DataFrame(
            {self.dt_col: pd.date_range(
                start = self.df[self.dt_col].min().date(),
                end = self.df[self.dt_col].max().date(), 
                freq='D')})\
        .reset_index().rename(columns={'index':'dt_index'})\
        .set_index(self.dt_col)
        
        # combine index and original data
        self.df = self.df.merge(self.date_index, left_on = self.dt_col, right_index = True, how='left')
        
    def get_cases(self, date_indeces):
        
        """
        Yields the reduced dataset containing cases of specified date indeces.
        -------------
        date_indeces: list of numerical date indeces
        """
        
        return self.df[self.df.dt_index.isin(date_indeces)]
        
    def get_baseline_cases(self):
        
        """
        Retrieves the cases corresponding to the baseline dates.
        """
        
        self.ref_df = self.get_cases(self.today - self.date_diff)
        self.n_ref_cases = len(self.ref_df) # total number of cases
    
    def get_current_cases(self):
        
        """
        Retrieves cases corresponding to current day.
        """
        
        self.curr_df = self.get_cases([self.today])
        self.n_curr_cases = len(self.curr_df) # number of cases from current day
        
    def calc_compensated_pval(self, score):
        
        """
        Calculates the compensated p-value using a randomization test of Fisher's Exact self.
        Uses racing to calculate when compensated p value likely will/will not meet user-defined
        alpha value to terminate racing early. Racing is evaluated every 10 simulations.
        """
        
        # combine current and baseline set
        cases_df = pd.concat([self.curr_df, self.ref_df], axis=0).copy().reset_index(drop=True)
        
        pvals = np.zeros(self.n_rands) # to store best scores from randomized sets
        comp_pvals = np.zeros(self.n_rands) # to store compensated p values
        
        # burn in randomizations
        for step in range(100):
                
            # shuffle cases (process equivalent to shuffling dates)
            cases_shuffled_df = cases_df.reindex(self.rng.permutation(cases_df.index))
            
            # separate baseline from current cases
            randomization_cases_df = cases_shuffled_df[-self.n_curr_cases:]
            randomization_reference_df = cases_shuffled_df[:-self.n_curr_cases]
            
            # calculate p value of best single score rule
            _, __, pvals[step] = self.best_single_rule(curr_df = randomization_cases_df,
                                                       ref_df = randomization_reference_df)
            
            # calculate compensated p value
            comp_pvals[step] = np.mean(pvals[:step + 1] < score)
            
        # begin race
        for step in range(100, self.n_rands):
            
            # check race every ten randomizations
            if step % 10 == 0:
                cp_sd = np.nanstd(comp_pvals[:step]) # standard deviation of compensated p values
                cp_mu = np.nanmean(comp_pvals[:step]) # average compensated p value
                
                # if highly significant with high likelihood
                if (cp_sd == 0) & (cp_mu == 0):
                    return comp_pvals[step - 1]
                
                # calculate 99% confidence interval
                cp_ub = comp_pvals[step - 1] + 2.807*cp_sd/np.sqrt(step) # upper bound of confidence interval
                cp_lb = comp_pvals[step - 1] - 2.807*cp_sd/np.sqrt(step) # lower bound of confidence interval
                
                # if not significant with high likelihood
                if cp_lb > self.alpha:
                    return None
                
                # if significant with high likelihood
                # if cp_ub <= self.alpha:
                #     return comp_pvals[step - 1]
                
            # if none of the conditions are met, continue race
                  
            # shuffle cases
            cases_shuffled_df = cases_df.reindex(self.rng.permutation(cases_df.index))
            
            # separate baseline from current cases
            randomization_cases_df = cases_shuffled_df[-self.n_curr_cases:]
            randomization_reference_df = cases_shuffled_df[:-self.n_curr_cases]
            
            # calculate p value of best single score rule
            _, __, pvals[step] = self.best_single_rule(curr_df = randomization_cases_df,
                                                   ref_df = randomization_reference_df)
            
            # calculate compensated p value
            comp_pvals[step] = np.mean(pvals[:step + 1] < score)            
            
        compensated_pval = np.mean(pvals <= score)
            
        if compensated_pval <= self.alpha:
            return compensated_pval

        else:
            return None
        
    def best_single_rule(self, curr_df, ref_df):
        
        """
        Computes minimum p-value for all one-rule combinations.
        """
        
        sig_pvals = []
        sig_feat_vals = []
        
        # iterate through feature columns to find best score for each feature
        if len(self.remaining_ftcols) > 0:

            for self.col in self.remaining_ftcols:
        
                # distribution by feature column in reference/baseline cases
                #ref_dist = ref_df.groupby(self.col).size().reset_index().rename(columns={0:'n'})
                ref_dist = ref_df.groupby(self.col).agg({'dt_index':'count'}).rename(columns={'dt_index':'n'})
                ref_dist['N'] = self.n_ref_cases - ref_dist['n'] # difference from total number - 
                                                                 # criticial for n_component implementation

                # distribution by feature column in current cases
                #curr_dist = curr_df.groupby(self.col).size().reset_index().rename(columns={0:'n'})
                curr_dist = curr_df.groupby(self.col).agg({'dt_index':'count'}).rename(columns={'dt_index':'n'})
                curr_dist['N'] = self.n_curr_cases - curr_dist['n'] # difference from total number - 
                                                                    # criticial for n_component implementation

                # score
                feat_val, p_val = self.best_score(current = curr_dist, ref = ref_dist)
                
                sig_pvals.append(p_val)
                sig_feat_vals.append(feat_val)
            
            # find best score overall
            best_idx = sig_pvals.index(min(sig_pvals))
            best_pval = sig_pvals[best_idx]
            best_ftval = sig_feat_vals[best_idx]
            best_ftcol = self.remaining_ftcols[best_idx]
            
            return best_ftcol, best_ftval, best_pval

        else:

            return None, None, 1
        
    def best_score(self, current, ref):
        
        """
        Performs Fisher Exact test to test for significant strange events. Returns the p-values
        and feature values that meet significance threshold. 
        ---------
        current: dataframe of current distribution by a feature column.
        ref: dataframe of reference distribution by feature column.
        """
        
        feat_val = None
        min_p_val = 1
        
        for ft_val in self.ft_dict[self.col]:
            
            # construct 2x2 contingency table
            try:
                current_vals = current.loc[ft_val]

            except:
                current_vals = np.array([0, 0])

            try:
                ref_vals = ref.loc[ft_val]

            except:
                ref_vals = np.array([0, 0])
            
            table = np.array([current_vals, ref_vals]).T
            
            # Fisher's Exact test
            _, p_val = stats.fisher_exact(table, alternative='greater')
            
            # check for significance
            if (p_val <= self.alpha) & (p_val < min_p_val):
                
                min_p_val = p_val
                feat_val = ft_val
                
        return feat_val, min_p_val
        
    def search_one_component(self, curr, ref):
        
        """
        Searches for strange events. Only considers one-component rules.
        """
        
        sig_pvals = []
        sig_feat_vals = []
        
        # find best (most significant) score its corresponding feature and feature value
        best_ftcol, best_ftval, best_score = self.best_single_rule(curr_df = curr,
                                                                  ref_df = ref)

        # find best (most significant) score and its feature value
        if best_score < self.alpha:
            
            # if performing randomization test
            if self.randomization is not None:

                comp_pval = self.calc_compensated_pval(best_score)

                if comp_pval is not None:

                    return best_ftcol, best_ftval, best_score, comp_pval
                
                else:
                    
                    return None, None, None, None
                        
            # if not performing randomization test       
            else:     
            
                return best_ftcol, best_ftval, best_score, None
        else:
            
            return None, None, None, None

    def full_search_single_component(self):
        
        """
        Runs full search on each day in input df. Only searches for one component rule per day.
        Returns dataframe of dates with significantly strange events and the corresponding features.
        """
    
        # initialize
        self.index_dates()
        self.get_unique_feature_values()
        self.alpha = self.alpha_single
        
        outbreak_feats = []
        outbreak_scores = []
        outbreak_dates = []
        outbreak_pvals = []
        
        # copy of feature columns
        self.remaining_ftcols = self.ft_col.copy()
        
        # unique dates of self.df
        dates = self.df.dt_index.unique()
        dates.sort()
        
        # iterate through days
        for date in dates:
            
            self.today = date
            
            # get current and reference datasets
            self.get_current_cases()
            self.get_baseline_cases()
            
            # search one component rules
            ftcol, feat, score, pval = self.search_one_component(curr = self.curr_df,
                                                                     ref = self.ref_df)
            
            # save significant results
            if feat is not None:
                outbreak_feats.append(feat)
                outbreak_scores.append(score)
                outbreak_dates.append(date)
                outbreak_pvals.append(pval)
        
        return pd.merge(pd.DataFrame({'features': outbreak_feats,
                                      'scores': outbreak_scores,
                                      'dates': outbreak_dates,
                                      'p_value':outbreak_pvals}), 
                        self.date_index.reset_index(), left_on='dates', right_on='dt_index', how='left')\
        .drop(['dates', 'dt_index'], axis=1)
    
    def search_subset_single_component(self, dates):
        
        """
        Runs full search on each day in input df. Only searches for one component rule per day.
        Returns dataframe of dates with significantly strange events and the corresponding features.
        This method is used to support non-parallel application of WSARE_marketer_dynamic class.
        """
    
        # initialize
        self.remaining_ftcols = self.ft_col.copy()
        self.get_unique_feature_values()
        self.alpha = self.alpha_single
        
        outbreak_feats = []
        outbreak_scores = []
        outbreak_dates = []
        outbreak_pvals = []
        
        # unique dates of self.df
        dates.sort()
        
        # iterate through days
        for date in dates:
            
            self.today = date
            
            # get current and reference datasets
            self.get_current_cases()
            self.get_baseline_cases()
            
            # search one component rules
            ftcol, feat, score, pval = self.search_one_component(curr = self.curr_df,
                                                                     ref = self.ref_df)
            
            # save significant results
            if feat is not None:
                outbreak_feats.append(feat)
                outbreak_scores.append(score)
                outbreak_dates.append(date)
                outbreak_pvals.append(pval)
        
        return pd.merge(pd.DataFrame({'features': outbreak_feats,
                                      'scores': outbreak_scores,
                                      'dates': outbreak_dates,
                                      'p_value':outbreak_pvals}), 
                        self.date_index.reset_index(), left_on='dates', right_on='dt_index', how='left')\
        .drop(['dates', 'dt_index'], axis=1)
    
    def initialize_components(self):
        
        """
        Initializes list of components in rule for n-component search.
        """
        
        self.rule_cols = []
        self.rule_vals = []
        self.remaining_ftcols = self.ft_col.copy() # features under consideration
    
    def refine_datasets(self):
        
        """
        Refines baseline and current cases dataset according to rule components 
        """
        n_refinements = len(self.rule_cols)
        
        self.ref_df_refined = self.ref_df[self.ref_df[self.rule_cols[0]] == self.rule_vals[0]]
        self.curr_df_refined = self.curr_df[self.curr_df[self.rule_cols[0]] == self.rule_vals[0]]
        
        if n_refinements > 1:
            for idx in range(1, n_refinements):
                self.ref_df_refined = self.ref_df_refined\
                                        [self.ref_df_refined[self.rule_cols[idx]] == self.rule_vals[idx]]
                self.curr_df_refined = self.curr_df_refined\
                                        [self.curr_df_refined[self.rule_cols[idx]] == self.rule_vals[idx]]
    
    def significance_of_new_component(self, ftcol, ftval):
        
        """
        Performs two distinct Fisher's exact tests to test for the significance of adding
        the new component to the existing set of components in the rule.
        """
        
        # get counts for current and baseline cases
        b1, o1, n1 = self.counts_for_double_test(self.curr_df, ftcol, ftval)
        b2, o2, n2 = self.counts_for_double_test(self.ref_df, ftcol, ftval)
        
        # first test
        table1 = np.array([[b1, b2],[n1, n2]])
        _, pval1 = stats.fisher_exact(table1, alternative='greater')
        
        # second test
        table2 = np.array([[b1, b2],[o1, o2]])
        _, pval2 = stats.fisher_exact(table2, alternative='greater')
        
        if max([pval1, pval2]) > self.alpha:
            return False
        else:
            return max([pval1, pval2])
        
    def counts_for_double_test(self, df, ftcol, ftval):
    
        """
        Returns counts from df according to old set of components and new component.
        """

        df = df.copy()

        # new component only
        new = df[df[ftcol] == ftval]

        # old component(s) only
        old = df

        # both sets of components
        both = new.copy()

        for idx in range(len(self.rule_cols)):

            col = self.rule_cols[idx]
            val = self.rule_vals[idx]

            both = both[both[col] == val]
            old = old[old[col] == val]

        return len(both), len(old), len(new)
    
    def create_results_dataframe(self):
        
        """
        Creates results dataframe.
        """
        
        self.results = pd.DataFrame(columns = self.ft_col + ['score', 'p_value', 'p_value_n', 'dt_index'])
    
    def update_results(self, score, pval, pval_n):
        
        """
        Updates results dataframe.
        """
        
        # save rule components
        for idx in range(len(self.rule_cols)):
            self.results.loc[self.results_row, self.rule_cols[idx]] = self.rule_vals[idx]

        # save rule scores
        self.results.loc[self.results_row, ['score', 'p_value', 'p_value_n', 'dt_index']] = [score, pval, pval_n, self.today]
        
    def full_search_n_component(self, n):
        
        """
        Runs full search on each day in input df. Considers multiple features in each rule.
        Each rule can have at most one value from each feature. Returns dataframe of dates 
        with significantly strange events and the corresponding features.
        """

        if n > len(self.ft_col):
            raise ValueError('Error: n must be <= number of feature columns.')

        # initialize
        self.index_dates()
        self.create_results_dataframe()

        self.results_row = 0
        keep_score = 1
        keep_pval = 1
        keep_pval_n = None

        # unique dates of self.df
        dates = self.df.dt_index.unique()
        dates.sort()

        # iterate through days
        for date in dates:

            self.today = date

            # initialize components
            self.initialize_components()
            self.get_unique_feature_values()
            self.alpha = self.alpha_single

            # get current and reference datasets
            self.get_current_cases()
            self.get_baseline_cases()

            # search one component rules
            ftcol, feat, score, pval = self.search_one_component(curr = self.curr_df,
                                                                 ref = self.ref_df)

            # save component
            self.rule_cols.append(ftcol)
            self.rule_vals.append(feat)

            # save scores
            keep_score = score
            keep_pval = pval
            keep_pval_n = None
            
            # if one component rule is detected
            if feat is not None:

                # refine datasets
                self.refine_datasets()
                
                # change alpha value
                self.alpha = self.alpha_n

                # remove significant feature column from future tests
                self.remaining_ftcols.remove(ftcol)

                # iterate through remaining feature columns
                while (len(self.remaining_ftcols) > 0) & (len(self.rule_cols) < n):

                    # search one component rules on refined datasets
                    ftcol_2, feat_2, score_2, pval_2 = self.search_one_component(curr = self.curr_df_refined,
                                                                                 ref = self.ref_df_refined)
                    
                    # if no additional significant feature
                    if feat_2 is not None:
                        
                        # test significance of adding new component to rule
                        pval_3 = self.significance_of_new_component(ftcol = ftcol_2,
                                                                  ftval = feat_2)

                        if pval_3:

                            # store new component
                            self.rule_cols.append(ftcol_2)
                            self.rule_vals.append(feat_2)

                            # store scores
                            keep_score = score_2
                            keep_pval = pval_2
                            keep_pval_n = pval_3

                            # refine datasets further
                            self.refine_datasets()

                            # remove feature column from further consideration
                            self.remaining_ftcols.remove(ftcol_2)

                        else:
                            break
                            
                    else:
                        break

                # save results
                self.update_results(keep_score, keep_pval, keep_pval_n)
                
                # update row
                self.results_row += 1

        return pd.merge(self.results,
                        self.date_index.reset_index(),
                        on='dt_index',
                        how='left')\
                    .drop(['dt_index'], axis=1) # with date values

    def search_subset_n_component(self, n, dates):
        
        """
        Runs full search on each day in input df. Considers multiple features in each rule.
        Each rule can have at most one value from each feature. Returns dataframe of dates 
        with significantly strange events and the corresponding features.
        """

        if n > len(self.ft_col):
            raise ValueError('Error: n must be <= number of feature columns.')

        # initialize
        self.create_results_dataframe()

        self.results_row = 0
        keep_score = 1
        keep_pval = 1
        keep_pval_n = None

        # unique dates of self.df
        dates.sort()

        # iterate through days
        for date in dates:

            self.today = date

            # initialize components
            self.initialize_components()
            self.get_unique_feature_values()
            self.alpha = self.alpha_single

            # get current and reference datasets
            self.get_current_cases()
            self.get_baseline_cases()

            # search one component rules
            ftcol, feat, score, pval = self.search_one_component(curr = self.curr_df,
                                                                 ref = self.ref_df)

            # save component
            self.rule_cols.append(ftcol)
            self.rule_vals.append(feat)

            # save scores
            keep_score = score
            keep_pval = pval
            keep_pval_n = None
            
            # if one component rule is detected
            if feat is not None:

                # refine datasets
                self.refine_datasets()
                
                # change alpha value
                self.alpha = self.alpha_n

                # remove significant feature column from future tests
                self.remaining_ftcols.remove(ftcol)

                # iterate through remaining feature columns
                while (len(self.remaining_ftcols) > 0) & (len(self.rule_cols) < n):

                    # search one component rules on refined datasets
                    ftcol_2, feat_2, score_2, pval_2 = self.search_one_component(curr = self.curr_df_refined,
                                                                                 ref = self.ref_df_refined)
                    
                    # if no additional significant feature
                    if feat_2 is not None:
                        
                        # test significance of adding new component to rule
                        pval_3 = self.significance_of_new_component(ftcol = ftcol_2,
                                                                  ftval = feat_2)

                        if pval_3:

                            # store new component
                            self.rule_cols.append(ftcol_2)
                            self.rule_vals.append(feat_2)

                            # store scores
                            keep_score = score_2
                            keep_pval = pval_2
                            keep_pval_n = pval_3

                            # refine datasets further
                            self.refine_datasets()

                            # remove feature column from further consideration
                            self.remaining_ftcols.remove(ftcol_2)

                        else:
                            break
                            
                    else:
                        break

                # save results
                self.update_results(keep_score, keep_pval, keep_pval_n)
                
                # update row
                self.results_row += 1

        return pd.merge(self.results,
                        self.date_index.reset_index(),
                        on='dt_index',
                        how='left')\
                    .drop(['dt_index'], axis=1) # with date values
    
    def full_search_single_component_parallel(self, n_cores):
        
        """
        Parallelized version of self.full_search_single_component.
        """
        
        # initialize dates
        self.index_dates()

        # unique dates of self.df
        dates = self.df.dt_index.unique()
        dates.sort()
        
        # intialize pool object
        pool = Pool(processes = n_cores)
        
        # execute
        results = pool.map_async(partial(call_WSARE_one_step_singlecomponent,
                                             self.df, # date-indexed dataset
                                             self.ft_col, # feature columns
                                             self.date_diff, # baseline days
                                             self.alpha_single, # alpha for first component
                                             self.randomization, # if randomizing
                                             self.n_rands), # number of randomizations
                                             dates).get() # unique date indices
        # close object
        pool.close()
        pool.join()
        
        # format and return results
        ans = []
        for result in results:
            if result is not None:
                ans.append(result)
            
        try:
            return pd.merge(pd.concat(ans).reset_index(drop=True),
                        self.date_index.reset_index(),
                        left_on='dates', right_on='dt_index',
                        how='left')\
                    .drop(['dt_index', 'dates'], axis=1) # with date values
        except:
            return
    
    def full_search_n_component_parallel(self, n, n_cores):
        
        """
        Parallelized version of self.full_search_n_component.
        """
        
        # initialize dates
        self.index_dates()

        # unique dates of self.df
        dates = self.df.dt_index.unique()
        dates.sort()
        
        # intialize pool object
        pool = Pool(processes = n_cores)
        
        # execute
        results = pool.map_async(partial(call_WSARE_one_step_ncomponent,
                                             self.df, # date-indexed dataset
                                             self.ft_col, # feature columns
                                             self.date_diff, # baseline days
                                             n, # maximum number of components in search
                                             self.alpha_single, # alpha for first component
                                             self.alpha_n, # alpha for second+ components
                                             self.randomization, # if randomizing
                                             self.n_rands), # number of randomizations
                                     dates).get() # unique date indices
        # close object
        pool.close()
        pool.join()
        
        # format and return results
        ans = []
        for result in results:
            if result is not None:
                ans.append(result)
            
        try:
            return pd.merge(pd.concat(ans).reset_index(drop=True),
                        self.date_index.reset_index(),
                        on='dt_index',
                        how='left')\
                    .drop(['dt_index'], axis=1) # with date values
        except:
            return


class WSARE_dynamic_PK(WSARE):

    """
    WSARE algorithm adjusted to consider fluctuating feature generalization.
    Contains methods for two differing approaches:
    1) Evaluating each day according to the most generalized form of the feature values
    across both the current day's cases and the baseline days' cases.
    2) Uniformly distributing the generalized case counts across the raw bins. This
    approach requires full_ref_pop to be provided.
    3) Evaluating each day according to the population joint distributions.
    """

    def __init__(self, df, feature_columns, date_column,
                 hierarchy, policy_codes,
                 count_column = None,
                 baseline_days = [56, 49, 42, 35],
                 alpha_single = 0.05, alpha_n = 0.05,
                 randomization = None, n_randomizations = 100,
                 full_ref_pop = None, pop_ref = None):
        
        self.df = df.copy()
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.df = self.df.sort_values(date_column).reset_index(drop=True)
        self.ft_col = feature_columns
        self.dt_col = date_column
        self.ct_col = count_column
        self.alpha_single = alpha_single
        self.alpha_n = alpha_n
        self.alpha = alpha_single
        self.date_diff = np.array(baseline_days)
        self.randomization = randomization
        self.n_rands = n_randomizations
        self.hierarchy = hierarchy
        self.policy_codes = policy_codes
        self.full_ref_pop = full_ref_pop
        self.pop_ref = pop_ref
        
        if randomization is not None:
            self.rng = np.random.default_rng()
            if n_randomizations < 100: 
                raise ValueError("Increase number of randomizations to >= 100 or set randomization = False")

    def remove_hidden_records(self):
        
        """
        Reduces dataset to days where data is shared, i.e., when policy index is not -1
        """
        
        self.df = self.df[self.df['policy'] > -1]

    def full_search_single_component_PK_dynamic_mostgeneral(self, n_cores):
        
        """
        Parallelized single component search when dataset is dynamically generalized according to the PK risk.
        This takes the approach of transforming all current day and baseline data to the most general form,
        for each set of current days and baseline days.
        """
        
        # reduce dataset to days where data is shared
        self.remove_hidden_records()
        
        # initialize dates
        self.index_dates()

        # unique dates of self.df
        dates = self.df.dt_index.unique()
        dates.sort()
        
        # intialize pool object
        pool = Pool(processes = n_cores)
        
        # execute
        results = pool.map_async(partial(call_WSARE_one_step_singlecomponent_dynamic_PK_mostgeneral,
                                             self.df.copy(), # date-indexed dataset
                                             self.ft_col, # feature columns
                                             self.hierarchy, # generalization hierarchy
                                             self.policy_codes, # policy codes
                                             self.date_diff, # baseline days
                                             self.alpha_single, # alpha for first component
                                             self.randomization, # if randomizing
                                             self.n_rands), # number of randomizations
                                             dates).get() # unique date indices

        # close object
        pool.close()
        pool.join()
        
        # format and return results
        ans = []
        for result in results:
            if result is not None:
                ans.append(result)
            
        try:
            return pd.merge(pd.concat(ans).reset_index(drop=True),
                        self.date_index.reset_index(),
                        left_on='dates', right_on='dt_index',
                        how='left')\
                    .drop(['dt_index', 'dates'], axis=1) # with date values
        except:
            return
    
    def full_search_n_component_PK_dynamic_mostgeneral(self, n, n_cores):
        
        """
        Parallelized n component search when dataset is dynamically generalized according to the PK risk.
        This takes the approach of transforming all current day and baseline data to the most general form,
        for each set of current days and baseline days.
        """
        
        # reduce dataset to days where data is shared
        self.remove_hidden_records()
        
        # initialize dates
        self.index_dates()

        # unique dates of self.df
        dates = self.df.dt_index.unique()
        dates.sort()
        
        # intialize pool object
        pool = Pool(processes = n_cores)
        
        # execute
        results = pool.map_async(partial(call_WSARE_one_step_ncomponent_dynamic_PK_mostgeneral,
                                             self.df, # date-indexed dataset
                                             self.ft_col, # feature columns
                                             self.hierarchy, # generalization hierarchy
                                             self.policy_codes, # policy codes
                                             self.date_diff, # baseline days
                                             n, # maximum number of components in search
                                             self.alpha_single, # alpha for first component
                                             self.alpha_n, # alpha for second+ components
                                             self.randomization, # if randomizing
                                             self.n_rands), # number of randomizations
                                     dates).get() # unique date indices
        
        # close object
        pool.close()
        pool.join()
        
        # format and return results
        ans = []
        for result in results:
            if result is not None:
                ans.append(result)
            
        try:
            return pd.merge(pd.concat(ans).reset_index(drop=True),
                        self.date_index.reset_index(),
                        on='dt_index',
                        how='left')\
                    .drop(['dt_index'], axis=1) # with date values
        except:
            return

    def add_age_group(self):

        if 'age_group' not in self.df.columns:
            self.df['age_group'] = pd.cut(self.df['age'].astype(int), [0, 10, 20, 30, 40, 50, 60, 70, 80, 120], right=False).astype(str)

    def full_search_n_component_PK_dynamic_uniform(self, n, n_cores):
        
        """
        Parallelized n component search when dataset is dynamically generalized according to the PK risk.
        This takes the approach of uniformly distributing the generalized case counts across the raw bins.
        This requires that the full reference population exists (i.e., the user inputs the census joint statistics).
        """

        # checks for necessary parameters
        if self.full_ref_pop is None:
            raise ValueError('No reference population provided')
        
        if self.ct_col is None:
            raise ValueError('No count column provided')

        # reduce dataset to days where data is shared
        self.remove_hidden_records()
        
        # initialize dates
        self.index_dates()

        # unique dates of self.df
        dates = self.df.dt_index.unique()
        dates.sort()

        # add age group
        self.add_age_group()
        
        # intialize pool object
        pool = Pool(processes = n_cores)
        
        # execute
        results = pool.map_async(partial(call_WSARE_one_step_ncomponent_dynamic_PK_uniform,
                                             self.df, # date-indexed dataset
                                             self.ft_col, # feature columns
                                             self.hierarchy, # generalization hierarchy
                                             self.ct_col, # count column
                                             self.dt_col, # date column
                                             self.full_ref_pop, # full reference population
                                             self.policy_codes, # policy codes
                                             self.date_diff, # baseline days
                                             n, # maximum number of components in search
                                             self.alpha_single, # alpha for first component
                                             self.alpha_n, # alpha for second+ components
                                             self.randomization, # if randomizing
                                             self.n_rands), # number of randomizations
                                     dates).get() # unique date indices
        
        # close object
        pool.close()
        pool.join()
        
        # format and return results
        ans = []
        for result in results:
            if result is not None:
                ans.append(result)
            
        try:
            return pd.merge(pd.concat(ans).reset_index(drop=True),
                        self.date_index.reset_index(),
                        on='dt_index',
                        how='left')\
                    .drop(['dt_index'], axis=1) # with date values
        except:
            return

    # def full_search_single_component_PK_dynamic_uniform(self, n_cores): <- Needs Fix
        
    #     """
    #     Parallelized single component search when dataset is dynamically generalized according to the PK risk.
    #     This takes the approach of uniformly distributing the generalized case counts across the raw bins.
    #     This requires that the full reference population exists (i.e., the user inputs the census joint statistics,
    #     with the same dates in the dataset).
    #     """

    #     # checks for necessary parameters
    #     if self.full_ref_pop is None:
    #         raise ValueError('No reference population provided')
    #         return
        
    #     if self.ct_col is None:
    #         raise ValueError('No count column provided')

    #     # uniformly distribute generalized case counts
    #     self.distribute_counts()

    #     # reduce dataset to days where data is shared
    #     self.remove_hidden_records_distributed()
        
    #     # initialize dates
    #     self.index_dates_distributed()

    #     # add age group
    #     self.add_age_group()

    #     # unique dates of self.df
    #     dates = self.distributed_df.dt_index.unique()
    #     dates.sort()
        
    #     # intialize pool object
    #     pool = Pool(processes = n_cores)
        
    #     # execute
    #     results = pool.map_async(partial(call_WSARE_one_step_singlecomponent_dynamic_PK_uniform,
    #                                          self.distributed_df, # date-indexed dataset
    #                                          self.ft_col, # feature columns
    #                                          self.ct_col, # count column
    #                                          self.date_diff, # baseline days
    #                                          self.alpha_single, # alpha for first component
    #                                          self.randomization, # if randomizing
    #                                          self.n_rands), # number of randomizations
    #                                          dates).get() # unique date indices

    #     # close object
    #     pool.close()
    #     pool.join()
        
    #     # format and return results
    #     ans = []
    #     for result in results:
    #         if result is not None:
    #             ans.append(result)
            
    #     try:
    #         return pd.merge(pd.concat(ans).reset_index(drop=True),
    #                     self.date_index.reset_index(),
    #                     left_on='dates', right_on='dt_index',
    #                     how='left')\
    #                 .drop(['dt_index', 'dates'], axis=1) # with date values
    #     except:
    #         return

    def standardize_generalization_popref(self, df):

        """
        Standardizes the generalization of the input df. Used in PK_dynamic_popref methods.
        """
    
        df = df.copy()

        age, race, sex, eth = self.policy_codes[self.pol]
        
        # age group 
        if age < 3:
            df['age_group'] = pd.cut(df['age'],
                                          self.hierarchy[0][age],
                                          right=False).astype(str)
        else:
            df['age_group'] = pd.cut(df['age'],
                                          [0, 10, 20, 30, 40, 50, 60, 70, 80, 120],
                                          right=False).astype(str)

        # race
        races = self.hierarchy[1][race]
        if races:
            for race_gen in races:
                for key, value in race_gen.items():
                    for val in value:
                        df.loc[df.race == val, 'race'] = key  

        # sex
        if sex == 0:
            df['sex'] = 'both_sex'

        # ethnicity
        if eth == 0:
            df['ethnicity'] = 'both_ethnicity'
            
        return df

    def full_search_n_component_PK_dynamic_popref(self, n, n_cores):
        
        """
        Parallelized n component search when dataset is dynamically generalized according to the PK risk.
        This takes the approach of evaluating each day's cases to the underlying demographic joint distributions
        represented in the self.pop_ref dataset. This requires the reference population exists (i.e., the user 
        inputs the census joint statistics without dates).
        """

        # checks for necessary parameters
        if self.pop_ref is None:
            raise ValueError('No reference population provided. Provide pop_ref.')

        # initialize dates
        self.index_dates()

        # unique policies
        uniq_pols = self.df.policy.unique()

        # filter out -1 (don't share) policy
        uniq_pols = uniq_pols[uniq_pols > -1]

        # number of individuals in reference population
        n_population = len(self.pop_ref)

        # intialize pool object
        pool = Pool(processes = n_cores)

        results = []

        # apply WSARE by policy
        for self.pol in uniq_pols:

            # generalize subset of dataset corresponding to policy
            gen_df = self.standardize_generalization_popref(self.df[self.df.policy == self.pol]).reset_index(drop=True)

            # generalize population reference dataset per policy parameters
            gen_popref = self.standardize_generalization_popref(self.pop_ref)

            # dates for subset of dataset
            dates = gen_df.dt_index.unique()

            # execute
            result = pool.map_async(partial(call_WSARE_one_step_ncomponent_dynamic_PK_popref,
                                             gen_df, # generalized, date-indexed dataset
                                             gen_popref, # generalized population reference dataset
                                             n_population, # number of individuals in population reference
                                             self.ft_col, # feature columns
                                             n, # maximum number of components in search
                                             self.alpha_single, # alpha for first component
                                             self.alpha_n, # alpha for second+ components
                                             self.randomization, # if randomizing
                                             self.n_rands), # number of randomizations
                                     dates).get() # unique date indices

            # add results
            for res in result:
                if res is not None:
                    results.append(res)

        # close object
        pool.close()
        pool.join()
        
        # format and return results
        try:
            return pd.merge(pd.concat(results).reset_index(drop=True),
                        self.date_index.reset_index(),
                        on='dt_index', how='left')\
                    .drop(['dt_index'], axis=1).sort_values('date') # with date values
        except:
            return

    def full_search_single_component_PK_dynamic_popref(self, n_cores):
        
        """
        Parallelized single component search when dataset is dynamically generalized according to the PK risk.
        This takes the approach of evaluating each day's cases to the underlying demographic joint distributions
        represented in the self.pop_ref dataset. This requires the reference population exists (i.e., the user 
        inputs the census joint statistics without dates).
        """

        # checks for necessary parameters
        if self.pop_ref is None:
            raise ValueError('No reference population provided. Provide pop_ref.')

        # initialize dates
        self.index_dates()

        # unique policies
        uniq_pols = self.df.policy.unique()

        # filter out -1 (don't share) policy
        uniq_pols = uniq_pols[uniq_pols > -1]

        # number of individuals in reference population
        n_population = len(self.pop_ref)

        # intialize pool object
        pool = Pool(processes = n_cores)

        results = []

        # apply WSARE by policy
        for self.pol in uniq_pols:

            # generalize subset of dataset corresponding to policy
            gen_df = self.standardize_generalization_popref(self.df[self.df.policy == self.pol]).reset_index(drop=True)

            # generalize population reference dataset per policy parameters
            gen_popref = self.standardize_generalization_popref(self.pop_ref)

            # dates for subset of dataset
            dates = gen_df.dt_index.unique()

            # execute
            result = pool.map_async(partial(call_WSARE_one_step_singlecomponent_dynamic_PK_popref, #### <- needs fixing. Does not work!!!
                                             gen_df, # generalized, date-indexed dataset
                                             gen_popref, # generalized population reference dataset
                                             n_population, # number of individuals in population reference
                                             self.ft_col, # feature columns
                                             self.alpha_single, # alpha for first component
                                             self.alpha_n, # alpha for second+ components
                                             self.randomization, # if randomizing
                                             self.n_rands), # number of randomizations
                                     dates).get() # unique date indices

            # add results
            for res in result:
                if res is not None:
                    results.append(res)

        # close object
        pool.close()
        pool.join()
        
        # format and return results
        try:
            return pd.merge(pd.concat(results).reset_index(drop=True),
                        self.date_index.reset_index(),
                        on='dt_index', how='left')\
                    .drop(['dt_index'], axis=1).sort_values('date') # with date values
        except:
            return


class WSARE_dynamic_marketer(WSARE):

    """
    WSARE algorithm adjusted to consider changing feature generalization, of the entire dataset,
    over time.
    """

    def __init__(self, df, feature_columns, date_column, baseline_days = [56, 49, 42, 35],
                 hierarchy = None, policy_codes = None,
                 alpha_single = 0.05, alpha_n = 0.05,
                 randomization = None, n_randomizations = 100):
    
        self.df = df.copy()
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.df = self.df.sort_values(date_column).reset_index(drop=True)
        self.ft_col = feature_columns
        self.dt_col = date_column
        self.alpha_single = alpha_single
        self.alpha_n = alpha_n
        self.alpha = alpha_single
        self.date_diff = np.array(baseline_days)
        self.randomization = randomization
        self.n_rands = n_randomizations
        self.hierarchy = hierarchy
        self.policy_codes = policy_codes

    def standardize_generalization(self, df):

        """
        Standardizes the generalization of the input df.
        """
    
        df = df.copy()

        age, race, sex, eth = self.policy_codes[self.pol]
        
        # age group 
        if age < 3:
            df['age_group'] = pd.cut(df['age'],
                                          self.hierarchy[0][age],
                                          right=False).astype(str)
        else:
            df['age_group'] = pd.cut(df['age'],
                                          [0, 10, 20, 30, 40, 50, 60, 70, 80, 120],
                                          right=False).astype(str)

        # race
        races = self.hierarchy[1][race]
        if races:
            for race_gen in races:
                for key, value in race_gen.items():
                    for val in value:
                        df.loc[df.race == val, 'race'] = key  

        # sex
        if sex == 0:
            df['sex'] = 'both_sex'

        # ethnicity
        if eth == 0:
            df['ethnicity'] = 'both_ethnicity'
            
        return df

    def full_search_single_component_marketer_dynamic_parallel(self, n_cores):
        
        """
        Parallelized single component search when dataset is dynamically generalized according to the marketer risk.
        """
        
        # initialize dates
        self.index_dates()

        # unique policies
        uniq_pols = self.df.policy.unique()

        # filter out -1 (don't share) policy
        uniq_pols = uniq_pols[uniq_pols > -1]

        # intialize pool object
        pool = Pool(processes = n_cores)

        results = []

        # apply WSARE by policy
        for self.pol in uniq_pols:

            # generalize dataset according to parameters of corresponding policy
            gen_df = self.standardize_generalization(self.df).reset_index(drop=True)

            # dates for subset of dataset corresponding to policy
            dates = gen_df[gen_df.policy == self.pol].dt_index.unique()

            # execute
            result = pool.map_async(partial(call_WSARE_one_step_singlecomponent,
                                                 gen_df, # generalized_date-indexed dataset
                                                 self.ft_col, # feature columns
                                                 self.date_diff, # baseline days
                                                 self.alpha_single, # alpha for first component
                                                 self.randomization, # if randomizing
                                                 self.n_rands), # number of randomizations
                                                 dates).get() # unique date indices

            # add results
            for res in result:
                if res is not None:
                    results.append(res)

        # close object
        pool.close()
        pool.join()
        
        # format and return results
        try:
            return pd.merge(pd.concat(results).reset_index(drop=True),
                        self.date_index.reset_index(),
                        on='dt_index', how='left')\
                    .drop(['dt_index'], axis=1).sort_values('date') # with date values
        except:
            return

    def full_search_single_component_marketer_dynamic(self):
        
        """
        Non-parallelized single component search when dataset is dynamically generalized according to the marketer risk.
        """
        
        # initialize dates
        self.index_dates()

        # unique policies
        uniq_pols = self.df.policy.unique()

        # filter out -1 (don't share) policy
        uniq_pols = uniq_pols[uniq_pols > -1]

        # initialize results
        results = []

        # apply WSARE by policy
        for self.pol in uniq_pols:

            # generalize dataset according to parameters of corresponding policy
            gen_df = self.standardize_generalization(self.df).reset_index(drop=True)

            # dates for subset of dataset corresponding to policy
            dates = gen_df[gen_df.policy == self.pol].dt_index.unique()

            # execute
            pol_results = WSARE(df = gen_df, feature_columns = self.ft_col, date_column = self.dt_col,
                                alpha_single = self.alpha_single, alpha_n = self.alpha_n, randomization = self.randomization,
                                n_randomizations = self.n_rands)

            # share date index
            pol_results.date_index = self.date_index

            results.append(pol_results.search_subset_single_component(dates = dates))

        return pd.concat(results, ignore_index=True)

    def full_search_n_component_marketer_dynamic(self, n):
        
        """
        Non-parallelized n component search when dataset is dynamically generalized according to the marketer risk.
        """
        
        # initialize dates
        self.index_dates()

        # unique policies
        uniq_pols = self.df.policy.unique()

        # filter out -1 (don't share) policy
        uniq_pols = uniq_pols[uniq_pols > -1]

        # initialize results
        results = []

        # apply WSARE by policy
        for self.pol in uniq_pols:

            # generalize dataset according to parameters of corresponding policy
            gen_df = self.standardize_generalization(self.df).reset_index(drop=True)

            # dates for subset of dataset corresponding to policy
            dates = gen_df[gen_df.policy == self.pol].dt_index.unique()

            # execute
            pol_results = WSARE(df = gen_df, feature_columns = self.ft_col, date_column = self.dt_col,
                                alpha_single = self.alpha_single, alpha_n = self.alpha_n, randomization = self.randomization,
                                n_randomizations = self.n_rands)

            # share date index
            pol_results.date_index = self.date_index

            results.append(pol_results.search_subset_n_component(n = n, dates = dates))

        return pd.concat(results, ignore_index=True)
    
    def full_search_n_component_marketer_dynamic_parallel(self, n, n_cores):
        
        """
        Parallelized n component search when dataset is dynamically generalized according to the marketer risk.
        """
        
        # initialize dates
        self.index_dates()

        # unique policies
        uniq_pols = self.df.policy.unique()

        # filter out -1 (don't share) policy
        uniq_pols = uniq_pols[uniq_pols > -1]

        # intialize pool object
        pool = Pool(processes = n_cores)

        results = []

        # apply WSARE by policy
        for self.pol in uniq_pols:

            # generalize full dataset according to parameters of corresponding policy
            gen_df = self.standardize_generalization(self.df).reset_index(drop=True)

            # dates for subset of dataset corresponding to policy
            dates = gen_df[gen_df.policy == self.pol].dt_index.unique()

            # execute
            result = pool.map_async(partial(call_WSARE_one_step_ncomponent,
                                     gen_df, # generalized, date-indexed dataset
                                     self.ft_col, # feature columns
                                     self.date_diff, # baseline days
                                     n, # maximum number of components in search
                                     self.alpha_single, # alpha for first component
                                     self.alpha_n, # alpha for second+ components
                                     self.randomization, # if randomizing
                                     self.n_rands), # number of randomizations
                             dates).get() # unique date indices

            # add results
            for res in result:
                if res is not None:
                    results.append(res)

        # close object
        pool.close()
        pool.join()
        
        # format and return results
        try:
            return pd.merge(pd.concat(results).reset_index(drop=True),
                        self.date_index.reset_index(),
                        on='dt_index', how='left')\
                    .drop(['dt_index'], axis=1).sort_values('date') # with date values
        except:
            return


## helper functions for parallel processing

def call_WSARE_one_step_ncomponent(df, feature_columns, baseline_days, n, alpha_single,
                                   alpha_n, randomization, n_randomizations, date):
    
    obj = WSARE_one_step(df = df,
                         feature_columns = feature_columns,
                         date = date,
                         baseline_days = baseline_days,
                         alpha_single = alpha_single,
                         alpha_n = alpha_n,
                         randomization = randomization,
                         n_randomizations = n_randomizations)
    
    return obj.full_search_n_component(n = n)

def call_WSARE_one_step_singlecomponent(df, feature_columns, baseline_days, alpha_single,
                                        randomization, n_randomizations, date):
    
    obj = WSARE_one_step(df = df,
                         feature_columns = feature_columns,
                         date = date,
                         baseline_days = baseline_days,
                         alpha_single = alpha_single,
                         alpha_n = None,
                         randomization = randomization,
                         n_randomizations = n_randomizations)
    
    return obj.full_search_single_component()

def call_WSARE_one_step_ncomponent_dynamic_PK_mostgeneral(df, feature_columns, hierarchy, policy_codes,
                                           baseline_days, n, alpha_single,
                                           alpha_n, randomization, n_randomizations, date):
    
    obj = WSARE_one_step_dynamic_PK_mostgeneral(df = df,
                                 feature_columns = feature_columns,
                                 date = date,
                                 hierarchy = hierarchy,
                                 policy_codes = policy_codes,
                                 baseline_days = baseline_days,
                                 alpha_single = alpha_single,
                                 alpha_n = alpha_n,
                                 randomization = randomization,
                                 n_randomizations = n_randomizations)
    
    return obj.full_search_n_component(n = n)

def call_WSARE_one_step_singlecomponent_dynamic_PK_mostgeneral(df, feature_columns, hierarchy, policy_codes,
                                                baseline_days, alpha_single,
                                                randomization, n_randomizations, date):
    
    obj = WSARE_one_step_dynamic_PK_mostgeneral(df = df,
                                 feature_columns = feature_columns,
                                 hierarchy = hierarchy,
                                 policy_codes = policy_codes,
                                 date = date,
                                 baseline_days = baseline_days,
                                 alpha_single = alpha_single,
                                 alpha_n = None,
                                 randomization = randomization,
                                 n_randomizations = n_randomizations)
    
    return obj.full_search_single_component()

def call_WSARE_one_step_ncomponent_dynamic_PK_uniform(df, feature_columns, hierarchy,
                                                    count_column, date_column, full_ref_pop,
                                                    policy_codes,
                                                   baseline_days, n, alpha_single,
                                                   alpha_n, randomization, n_randomizations, date):
    
    obj = WSARE_one_step_dynamic_PK_uniform(df = df,
                                            feature_columns = feature_columns,
                                            hierarchy = hierarchy,
                                            count_column = count_column,
                                            date = date,
                                            date_column = date_column,
                                            full_ref_pop = full_ref_pop,
                                            policy_codes = policy_codes,
                                            baseline_days = baseline_days,
                                            alpha_single = alpha_single,
                                            alpha_n = alpha_n,
                                            randomization = randomization,
                                            n_randomizations = n_randomizations)
    
    return obj.full_search_n_component(n = n)

def call_WSARE_one_step_singlecomponent_dynamic_PK_uniform(df, feature_columns, count_column,
                                                baseline_days, alpha_single,
                                                randomization, n_randomizations, date):
    
    obj = WSARE_one_step_dynamic_PK_uniform(df = df,
                                            feature_columns = feature_columns,
                                            count_column = count_column,
                                            date = date,
                                            baseline_days = baseline_days,
                                            alpha_single = alpha_single,
                                            alpha_n = None,
                                            randomization = randomization,
                                            n_randomizations = n_randomizations)
    
    return obj.full_search_single_component()

def call_WSARE_one_step_ncomponent_dynamic_marketer(df, feature_columns, hierarchy, policy_codes,
                                           baseline_days, n, alpha_single,
                                           alpha_n, randomization, n_randomizations, date):
    
    obj = WSARE_one_step_dynamic_marketer(df = df,
                                 feature_columns = feature_columns,
                                 date = date,
                                 hierarchy = hierarchy,
                                 policy_codes = policy_codes,
                                 baseline_days = baseline_days,
                                 alpha_single = alpha_single,
                                 alpha_n = alpha_n,
                                 randomization = randomization,
                                 n_randomizations = n_randomizations)
    
    return obj.full_search_n_component(n = n)

def call_WSARE_one_step_singlecomponent_dynamic_marketer(df, feature_columns, hierarchy, policy_codes,
                                                baseline_days, alpha_single,
                                                randomization, n_randomizations, date):
    
    obj = WSARE_one_step_dynamic_marketer(df = df,
                                 feature_columns = feature_columns,
                                 hierarchy = hierarchy,
                                 policy_codes = policy_codes,
                                 date = date,
                                 baseline_days = baseline_days,
                                 alpha_single = alpha_single,
                                 alpha_n = None,
                                 randomization = randomization,
                                 n_randomizations = n_randomizations)
    
    return obj.full_search_single_component()

def call_WSARE_one_step_ncomponent_dynamic_PK_popref(df, popref, n_population,
                                                     feature_columns, n, alpha_single,
                                                     alpha_n, randomization,
                                                     n_randomizations, date):
    
    obj = WSARE_one_step_dynamic_PK_popref(df = df,
                                           pop_ref = popref,
                                           n_population = n_population,
                                           feature_columns = feature_columns,
                                           date = date,
                                           alpha_single = alpha_single,
                                           alpha_n = alpha_n,
                                           randomization = randomization,
                                           n_randomizations = n_randomizations)
    
    return obj.full_search_n_component(n = n)

def call_WSARE_one_step_singlecomponent_dynamic_PK_popref(df, popref, n_population,
                                                          feature_columns, alpha_single,
                                                          alpha_n, randomization,
                                                          n_randomizations, date):
    
    obj = WSARE_one_step_dynamic_PK_popref(df = df,
                                           pop_ref = popref,
                                           n_population = n_population,
                                           feature_columns = feature_columns,
                                           date = date,
                                           alpha_single = alpha_single,
                                           alpha_n = None,
                                           randomization = randomization,
                                           n_randomizations = n_randomizations)
    
    return obj.full_search_single_component()



## trimmed WSARE classes used for parallel processing
class WSARE_one_step(WSARE):
    
    def __init__(self, df, feature_columns, date, baseline_days = [56, 49, 42, 35],
                 alpha_single = 0.05, alpha_n = 0.05,
                 randomization = None, n_randomizations = 100):
        
        self.df = df.copy()
        self.ft_col = feature_columns
        self.alpha_single = alpha_single
        self.alpha_n = alpha_n
        self.alpha = alpha_single
        self.date_diff = np.array(baseline_days)
        self.randomization = randomization
        self.n_rands = n_randomizations
        self.today = date # day of evaluation
        self.results_row = 0
        
        if randomization is not None:
            self.rng = np.random.default_rng()
            if n_randomizations < 100: 
                raise ValueError("Increase number of randomizations to >= 100 or set randomization = False")

    def reduce_dataset(self):
        
        """
        This class only evaluates the strange events for one day, so this function
        reduces the full dataset to the current date and the baseline dates.
        """
        selected_days = np.append((self.today - self.date_diff), [self.today])

        self.df = self.df[self.df.dt_index.isin(selected_days)]

    def full_search_single_component(self):
        
        """
        Runs full search on each day in input df. Only searches for one component rule per day.
        Returns dataframe of dates with significantly strange events and the corresponding features.
        """

        # reduce dataset to days of interest
        self.reduce_dataset()
        
        # if no data is shared
        if len(self.df) == 0:
            return None

        # initialize
        self.alpha = self.alpha_single
        self.remaining_ftcols = self.ft_col.copy()
        self.get_unique_feature_values()
            
        # get current and reference datasets
        self.get_current_cases()
        self.get_baseline_cases()

        # search one component rules
        ftcol, feat, score, pval = self.search_one_component(curr = self.curr_df,
                                                             ref = self.ref_df)

        # save significant results
        if feat is not None:
            return pd.DataFrame({'features': [feat],
                                  'scores': [score],
                                  'dates': [self.today],
                                  'p_value': [pval]})
        else:
            return None
          
    def full_search_n_component(self, n):
        
        """
        Runs full search on each day in input df. Considers multiple features in each rule.
        Each rule can have at most one value from each feature. Returns dataframe of dates 
        with significantly strange events and the corresponding features.
        """

        if n > len(self.ft_col):
            raise ValueError('Error: n must be <= number of feature columns.')

        # reduce dataset to days of interest
        self.reduce_dataset()
        
        # if no data is shared
        if len(self.df) == 0:
            return None

        # initialize
        self.create_results_dataframe()
        self.initialize_components()
        self.get_unique_feature_values()

        keep_score = 1
        keep_pval = 1
        keep_pval_n = None

        self.alpha = self.alpha_single

        # get current and reference datasets
        self.get_current_cases()
        self.get_baseline_cases()

        # search one component rules
        ftcol, feat, score, pval = self.search_one_component(curr = self.curr_df,
                                                             ref = self.ref_df)

        # save component
        self.rule_cols.append(ftcol)
        self.rule_vals.append(feat)

        # save scores
        keep_score = score
        keep_pval = pval

        # if one component rule is detected
        if feat is not None:

            # refine datasets
            self.refine_datasets()

            # change alpha value
            self.alpha = self.alpha_n

            # remove significant feature column from future tests
            self.remaining_ftcols.remove(ftcol)

            # iterate through remaining feature columns
            while (len(self.remaining_ftcols) > 0) & (len(self.rule_cols) < n):

                # search one component rules on refined datasets
                ftcol_2, feat_2, score_2, pval_2 = self.search_one_component(curr = self.curr_df_refined,
                                                                             ref = self.ref_df_refined)

                # if no additional significant feature
                if feat_2 is not None:

                    # test significance of adding new component to rule
                    pval_3 = self.significance_of_new_component(ftcol = ftcol_2,
                                                              ftval = feat_2)

                    if pval_3:

                        # store new component
                        self.rule_cols.append(ftcol_2)
                        self.rule_vals.append(feat_2)

                        # store scores
                        keep_score = score_2
                        keep_pval = pval_2
                        keep_pval_n = pval_3

                        # refine datasets further
                        self.refine_datasets()

                        # remove feature column from further consideration
                        self.remaining_ftcols.remove(ftcol_2)

                    else:
                        break

                else:
                    break

            # save results
            self.update_results(keep_score, keep_pval, keep_pval_n)

            return self.results
        
        else:
            return None

class WSARE_one_step_dynamic_PK_mostgeneral(WSARE_one_step):
    
    def __init__(self, df, feature_columns, date, hierarchy,
                 policy_codes, baseline_days = [56, 49, 42, 35],
                 alpha_single = 0.05, alpha_n = 0.05,
                 randomization = None, n_randomizations = 100):
        
        self.df = df.copy()
        self.ft_col = feature_columns
        self.alpha_single = alpha_single
        self.alpha_n = alpha_n
        self.alpha = alpha_single
        self.date_diff = np.array(baseline_days)
        self.randomization = randomization
        self.n_rands = n_randomizations
        self.today = date # day of evaluation
        self.hier = hierarchy
        self.pol_codes = np.stack(policy_codes)
        self.results_row = 0
        
        if randomization is not None:
            self.rng = np.random.default_rng()
            if n_randomizations < 100: 
                raise ValueError("Increase number of randomizations to >= 100 or set randomization = False")
        
    def standardize_generalization(self):
        
        """
        Standardizes each day's case record generalization to the most general policy
        applied to either the current day or any of the baseline days.
        """
        # get unique policy indices
        uniq_policies = self.df.policy.unique()
        
        # get policy parameters
        gen_params = list(self.pol_codes[uniq_policies].min(axis=0))

        age, race, sex, ethn = gen_params

        # age group 
        if age < 3:
            self.df['age_group'] = pd.cut(self.df['age'],
                                          self.hier[0][age],
                                          right=False).astype(str)
        else:
            self.df['age_group'] = pd.cut(self.df['age'],
                                          [0, 10, 20, 30, 40, 50, 60, 70, 80, 120],
                                          right=False).astype(str)

        # race
        races = self.hier[1][race]
        if races:
            for race_gen in races:
                for key, value in race_gen.items():
                    for val in value:
                        self.df.loc[self.df.race == val, 'race'] = key  

        # sex
        if sex == 0:
            self.df['sex'] = 'both_sex'
            
        # ethnicity
        if ethn == 0:
            self.df['ethnicity'] = 'both_ethnicity'
    
    def full_search_single_component(self):
        
        """
        Runs full search on each day in input df. Only searches for one component rule per day.
        Returns dataframe of dates with significantly strange events and the corresponding features.
        """
    
        # initialize
        self.reduce_dataset()
        
        # if no data is shared
        if len(self.df) == 0:
            return None
        
        self.standardize_generalization()
        self.alpha = self.alpha_single
        
        # copy of feature columns
        self.remaining_ftcols = self.ft_col.copy()
        self.get_unique_feature_values()
            
        # get current and reference datasets
        self.get_current_cases()
        self.get_baseline_cases()

        # search one component rules
        ftcol, feat, score, pval = self.search_one_component(curr = self.curr_df,
                                                             ref = self.ref_df)

        # save significant results
        if feat is not None:
            return pd.DataFrame({'features': [feat],
                                  'scores': [score],
                                  'dates': [self.today],
                                  'p_value': [pval]})
        else:
            return None
    
    def full_search_n_component(self, n):
        
        """
        Runs full search on each day in input df. Considers multiple features in each rule.
        Each rule can have at most one value from each feature. Returns dataframe of dates 
        with significantly strange events and the corresponding features.
        """

        if n > len(self.ft_col):
            raise ValueError('Error: n must be <= number of feature columns.')

        # initialize
        self.reduce_dataset()
        
        # if no data is shared
        if len(self.df) == 0:
            return None
        
        self.standardize_generalization()
        self.create_results_dataframe()

        keep_score = 1
        keep_pval = 1
        keep_pval_n = None

        # initialize components
        self.initialize_components()
        self.get_unique_feature_values()
        self.alpha = self.alpha_single

        # get current and reference datasets
        self.get_current_cases()
        self.get_baseline_cases()

        # search one component rules
        ftcol, feat, score, pval = self.search_one_component(curr = self.curr_df,
                                                             ref = self.ref_df)

        # save component
        self.rule_cols.append(ftcol)
        self.rule_vals.append(feat)

        # save scores
        keep_score = score
        keep_pval = pval

        # if one component rule is detected
        if feat is not None:

            # refine datasets
            self.refine_datasets()

            # change alpha value
            self.alpha = self.alpha_n

            # remove significant feature column from future tests
            self.remaining_ftcols.remove(ftcol)

            # iterate through remaining feature columns
            while (len(self.remaining_ftcols) > 0) & (len(self.rule_cols) < n):

                # search one component rules on refined datasets
                ftcol_2, feat_2, score_2, pval_2 = self.search_one_component(curr = self.curr_df_refined,
                                                                             ref = self.ref_df_refined)

                # if no additional significant feature
                if feat_2 is not None:

                    # test significance of adding new component to rule
                    pval_3 = self.significance_of_new_component(ftcol = ftcol_2,
                                                              ftval = feat_2)

                    if pval_3:

                        # store new component
                        self.rule_cols.append(ftcol_2)
                        self.rule_vals.append(feat_2)

                        # store scores
                        keep_score = score_2
                        keep_pval = pval_2
                        keep_pval_n = pval_3

                        # refine datasets further
                        self.refine_datasets()

                        # remove feature column from further consideration
                        self.remaining_ftcols.remove(ftcol_2)

                    else:
                        break

                else:
                    break

            # save results
            self.update_results(keep_score, keep_pval, keep_pval_n)

            return self.results
        
        else:
            return None

class WSARE_one_step_dynamic_PK_uniform(WSARE_one_step):
    
    def __init__(self, df, feature_columns, hierarchy,
                 date, date_column, count_column, policy_codes,
                 full_ref_pop,
                 baseline_days = [56, 49, 42, 35],
                 alpha_single = 0.05, alpha_n = 0.05,
                 randomization = None, n_randomizations = 100):

        self.df = df.copy()
        self.ft_col = feature_columns
        self.dt_col = date_column
        self.ct_col = count_column
        self.alpha_single = alpha_single
        self.alpha_n = alpha_n
        self.alpha = alpha_single
        self.date_diff = np.array(baseline_days)
        self.randomization = randomization
        self.n_rands = n_randomizations
        self.hierarchy = hierarchy
        self.policy_codes = policy_codes
        self.full_ref_pop = full_ref_pop
        self.today = date # day of evaluation
        self.results_row = 0
        
        if randomization is not None:
            self.rng = np.random.default_rng()
            if n_randomizations < 100: 
                raise ValueError("Increase number of randomizations to >= 100 or set randomization = False")

    def get_unique_feature_values(self):
        
        """
        Stores a list of the unique set of values for each feature in feature_columns.
        """
        
        df = self.distribute_counts(self.curr_df)
        self.ft_dict={}

        for ft in self.ft_col:
            
            uniq_vals = df[ft].unique()
            
            # if there is only one unique value for a feature column,
            # remove the column from consideration
            if len(uniq_vals) == 1:
                
                self.remaining_ftcols.remove(ft)
                
            else:
                
                self.ft_dict[ft] = uniq_vals

    def reduce_dataset(self):
        
        """
        This class only evaluates the strange events for one day, so this function
        reduces the full dataset to the current date and the baseline dates.
        """

        # filter by days
        selected_days = np.append((self.today - self.date_diff), [self.today])

        self.df = self.df[self.df.dt_index.isin(selected_days)].astype({self.dt_col:str})
        self.full_ref_pop = self.full_ref_pop[self.full_ref_pop[self.dt_col].isin(self.df[self.dt_col].unique())]

    def distribute_counts(self, df):
    
        """
        Uniformly distributes generalized case counts in the simulated dataset across
        the raw bins. This function requires a reference population, which can be provided
        by the get_simulated_and_reference_populations_uniform function. It also requires a policy-indexed
        version of the simulated dataset, the list of corresponding policies, and the generalization
        hierarchy to which the policies correspond.
        """

        if len(df) == 0:
            df['deid_counts'] = None
            df['age_group'] = None
            return df

        qid = ['age', 'sex', 'race', 'ethnicity'] + [self.dt_col]
            
        # format date values
        PK_indexed_df = df.astype({self.dt_col:str})
        
        # counts per bin
        groups = PK_indexed_df.groupby(qid).size().reset_index().rename(columns = {0: 'counts'})

        # all potential groups according to reference population
        uniq_dates = groups[self.dt_col].unique()
        red_ref = self.full_ref_pop.loc[self.full_ref_pop[self.dt_col].isin(uniq_dates)]
        full_groups = pd.merge(red_ref, groups, on = qid, how = 'left').fillna(0)

        # add policy index
        groups_wpol = pd.merge(full_groups,
                               PK_indexed_df.drop(['race', 'sex', 'ethnicity', 'age', 'age_group'],
                                                  axis=1).drop_duplicates(),
                               on = 'date', how='inner')
        
        # unique policies in the dataset
        uniq_pols = PK_indexed_df['policy'].unique()

        # do not share -1 policy
        uniq_pols = uniq_pols[uniq_pols > -1]
        
        # average group counts across bins for each policy
        new_df = []

        for pol in uniq_pols:

            age, race, sex, eth = self.policy_codes[pol]

            sub_df = groups_wpol[groups_wpol['policy'] == pol]

            new_df.append(policy_dataset_counts(sub_df,
                                      ages = self.hierarchy[0][age],
                                      races = self.hierarchy[1][race],
                                      sexes = self.hierarchy[2][sex],
                                      ethnicities = self.hierarchy[3][eth],
                                      combined_race_ethnicity = False,
                                      months = False))
            
        # combine dataset pieces
        comb_df = pd.concat(new_df)
        
        # combine with original groups counts - this fills in zeros for dates where no data is shared
        distributed_df = pd.merge(groups_wpol, comb_df.drop(qid + ['counts', 'policy', 'dt_index'], axis=1),
                        left_index = True, right_index = True, how='left').fillna(0)

        # add age_group
        distributed_df['age_group'] = pd.cut(distributed_df['age'].astype(int),
                                                     [0, 10, 20, 30, 40, 50, 60, 70, 80, 120], right=False).astype(str)

        # filter rows with no data
        return distributed_df[distributed_df.deid_counts > 0]

    def search_one_component(self, curr, ref):
        
        """
        Searches for strange events. Only considers one-component rules.
        """
        
        sig_pvals = []
        sig_feat_vals = []
        
        # find best (most significant) score its corresponding feature and feature value
        best_ftcol, best_ftval, best_score = self.best_single_rule(curr_df = curr,
                                                                  ref_df = ref)

        # find best (most significant) score and its feature value
        if best_score < self.alpha:
            
            # if performing randomization test
            if self.randomization is not None:

                comp_pval = self.calc_compensated_pval(best_score)

                if comp_pval is not None:

                    return best_ftcol, best_ftval, best_score, comp_pval
                
                else:
                    
                    return None, None, None, None
                        
            # if not performing randomization test       
            else:     
            
                return best_ftcol, best_ftval, best_score, None
        else:
            
            return None, None, None, None
        
    def calc_compensated_pval(self, score):
        
        """
        Calculates the compensated p-value using a randomization test of Fisher's Exact self.
        Uses racing to calculate when compensated p value likely will/will not meet user-defined
        alpha value to terminate racing early. Racing is evaluated every 10 simulations.
        """
        
        # combine current and baseline set
        cases_df = pd.concat([self.curr_df, self.ref_df], axis=0).copy().reset_index(drop=True)
        
        pvals = np.zeros(self.n_rands) # to store best scores from randomized sets
        comp_pvals = np.zeros(self.n_rands) # to store compensated p values
        
        # burn in randomizations
        for step in range(100):
                
            # shuffle cases (process equivalent to shuffling dates)
            # without shuffling policy, self.dt_col, and dt_index column values.
            # This allows for uniform distribution by each day's policy.
            cases_shuffled_df = cases_df.reindex(self.rng.permutation(cases_df.index))
            cases_shuffled_df[['policy',self.dt_col, 'dt_index']] = cases_df[['policy',self.dt_col, 'dt_index']].values
            
            # separate baseline from current cases
            randomization_cases_df = cases_shuffled_df[:self.n_curr_cases]
            randomization_reference_df = cases_shuffled_df[self.n_curr_cases:]
            
            # calculate p value of best single score rule
            _, __, pvals[step] = self.best_single_rule(curr_df = self.distribute_counts(randomization_cases_df),
                                                   ref_df = self.distribute_counts(randomization_reference_df))
            
            # calculate compensated p value
            comp_pvals[step] = np.mean(pvals[:step + 1] < score)
            
        # begin race
        for step in range(100, self.n_rands):
            
            # check race every ten randomizations
            if step % 10 == 0:
                cp_sd = np.std(comp_pvals[:step]) # standard deviation of compensated p values
                cp_mu = np.mean(comp_pvals[:step]) # average compensated p value
                
                # # if highly significant with high likelihood
                # if (cp_sd == 0) & (cp_mu == 0):
                #     return comp_pvals[step - 1]
                
                # calculate 99% confidence interval
                cp_ub = comp_pvals[step - 1] + 2.576*cp_sd/np.sqrt(step) # upper bound of confidence interval
                cp_lb = comp_pvals[step - 1] - 2.576*cp_sd/np.sqrt(step) # lower bound of confidence interval
                
                # if not significant with high likelihood
                if cp_lb > self.alpha:
                    return None
                
                # # if significant with high likelihood
                # if cp_ub <= self.alpha:
                #     return comp_pvals[step - 1]
                
            # if none of the conditions are met, continue race
                  
            # shuffle cases
            cases_shuffled_df = cases_df.reindex(self.rng.permutation(cases_df.index))
            cases_shuffled_df[['policy',self.dt_col, 'dt_index']] = cases_df[['policy',self.dt_col, 'dt_index']].values
            
            # separate baseline from current cases
            randomization_cases_df = cases_shuffled_df[:self.n_curr_cases]
            randomization_reference_df = cases_shuffled_df[self.n_curr_cases:]
            
            # calculate p value of best single score rule
            _, __, pvals[step] = self.best_single_rule(curr_df = self.distribute_counts(randomization_cases_df),
                                                   ref_df = self.distribute_counts(randomization_reference_df))
            
            # calculate compensated p value
            comp_pvals[step] = np.mean(pvals[:step + 1] < score)            
            
        compensated_pval = np.mean(pvals <= score)
            
        if compensated_pval <= self.alpha:
            return compensated_pval

        else:
            return None
        
    def best_single_rule(self, curr_df, ref_df):
        
        """
        Computes minimum p-value for all one-rule combinations.
        """
        
        sig_pvals = []
        sig_feat_vals = []
        
        # iterate through feature columns to find best score for each feature
        if len(self.remaining_ftcols) > 0:

            for self.col in self.remaining_ftcols:
        
                # distribution by feature column in reference/baseline cases
                ref_dist = ref_df.groupby(self.col).agg({self.ct_col: 'sum'}).rename(columns={self.ct_col:'n'})
                ref_dist['N'] = self.n_ref_cases - ref_dist['n'] # difference from total number - 
                                                                 # criticial for n_component implementation

                # distribution by feature column in current cases
                curr_dist = curr_df.groupby(self.col).agg({self.ct_col: 'sum'}).rename(columns={self.ct_col:'n'})
                curr_dist['N'] = self.n_curr_cases - curr_dist['n'] # difference from total number - 
                                                                    # criticial for n_component implementation

                # score
                feat_val, p_val = self.best_score(current = curr_dist, ref = ref_dist)
                
                sig_pvals.append(p_val)
                sig_feat_vals.append(feat_val)
            
            # find best score overall
            best_idx = sig_pvals.index(min(sig_pvals))
            best_pval = sig_pvals[best_idx]
            best_ftval = sig_feat_vals[best_idx]
            best_ftcol = self.remaining_ftcols[best_idx]
            
            return best_ftcol, best_ftval, best_pval

        else:

            return None, None, 1
        
    def counts_for_double_test(self, df, ftcol, ftval):
    
        """
        Returns counts from df according to old set of components and new component.
        """

        df = df.copy()

        # new component only
        new = df[df[ftcol] == ftval]

        # old component(s) only
        old = df

        # both sets of components
        both = new.copy()

        for idx in range(len(self.rule_cols)):

            col = self.rule_cols[idx]
            val = self.rule_vals[idx]

            both = both[both[col] == val]
            old = old[old[col] == val]

        return both.deid_counts.sum(), old.deid_counts.sum(), new.deid_counts.sum()

    def significance_of_new_component(self, ftcol, ftval):
        
        """
        Performs two distinct Fisher's exact tests to test for the significance of adding
        the new component to the existing set of components in the rule.
        """
        
        # get counts for current and baseline cases
        b1, o1, n1 = self.counts_for_double_test(self.distribute_counts(self.curr_df), ftcol, ftval)
        b2, o2, n2 = self.counts_for_double_test(self.distribute_counts(self.ref_df), ftcol, ftval)
        
        # first test
        table1 = np.array([[b1, b2],[n1, n2]])
        _, pval1 = stats.fisher_exact(table1, alternative='greater')
        
        # second test
        table2 = np.array([[b1, b2],[o1, o2]])
        _, pval2 = stats.fisher_exact(table2, alternative='greater')
        
        if max([pval1, pval2]) > self.alpha:
            return False
        else:
            return max([pval1, pval2])

    def full_search_n_component(self, n):
        
        """
        Runs full search on each day in input df. Considers multiple features in each rule.
        Each rule can have at most one value from each feature. Returns dataframe of dates 
        with significantly strange events and the corresponding features.
        """

        if n > len(self.ft_col):
            raise ValueError('Error: n must be <= number of feature columns.')

        # initialize
        self.reduce_dataset()
        
        # if no data is shared
        if len(self.df) == 0:
            return None
        
        self.create_results_dataframe()

        keep_score = 1
        keep_pval = 1
        keep_pval_n = None

        # initialize components
        self.initialize_components()
        self.alpha = self.alpha_single

        # get current and reference datasets
        self.get_current_cases()
        self.get_baseline_cases()
        self.get_unique_feature_values()

        if len(self.curr_df) == 0:
            return None

        # search one component rules
        ftcol, feat, score, pval = self.search_one_component(curr = self.distribute_counts(self.curr_df),
                                                             ref = self.distribute_counts(self.ref_df))

        # save component
        self.rule_cols.append(ftcol)
        self.rule_vals.append(feat)

        # save scores
        keep_score = score
        keep_pval = pval

        # if one component rule is detected
        if feat is not None:

            # refine datasets
            self.refine_datasets()

            # change alpha value
            self.alpha = self.alpha_n

            # remove significant feature column from future tests
            self.remaining_ftcols.remove(ftcol)

            # iterate through remaining feature columns
            while (len(self.remaining_ftcols) > 0) & (len(self.rule_cols) < n):

                # search one component rules on refined datasets
                ftcol_2, feat_2, score_2, pval_2 = self.search_one_component(curr = self.distribute_counts(self.curr_df_refined),
                                                                             ref = self.distribute_counts(self.ref_df_refined))

                # if no additional significant feature
                if feat_2 is not None:

                    # test significance of adding new component to rule
                    pval_3 = self.significance_of_new_component(ftcol = ftcol_2,
                                                              ftval = feat_2)

                    if pval_3:

                        # store new component
                        self.rule_cols.append(ftcol_2)
                        self.rule_vals.append(feat_2)

                        # store scores
                        keep_score = score_2
                        keep_pval = pval_2
                        keep_pval_n = pval_3

                        # refine datasets further
                        self.refine_datasets()

                        # remove feature column from further consideration
                        self.remaining_ftcols.remove(ftcol_2)

                    else:
                        break

                else:
                    break

            # save results
            self.update_results(keep_score, keep_pval, keep_pval_n)

            return self.results
        
        else:
            return None

class WSARE_one_step_dynamic_PK_popref(WSARE_one_step):
    
    def __init__(self, df, pop_ref, n_population,
                 feature_columns, date,
                 alpha_single = 0.05, alpha_n = 0.05,
                 randomization = None, n_randomizations = 100):
        
        self.df = df.copy() # already generalized
        self.ref_df = pop_ref.copy() # already generalized
        self.n_ref_cases = n_population # number of individuals in reference population
        self.ft_col = feature_columns
        self.alpha_single = alpha_single
        self.alpha_n = alpha_n
        self.randomization = randomization
        self.n_rands = n_randomizations
        self.alpha = alpha_single
        self.today = date # day of evaluation
        self.results_row = 0

        if randomization is not None:
            self.rng = np.random.default_rng()
            if n_randomizations < 100: 
                raise ValueError("Increase number of randomizations to >= 100 or set randomization = False")

    def reduce_dataset(self):
        
        """
        This class only evaluates the strange events for one day, so this function
        reduces the full dataset to the current date (No baseline dates because the
        reference population is used as the baseline).
        """
        
        self.df = self.df[self.df.dt_index == self.today]

    def full_search_single_component(self):
        
        """
        Runs full search on each day in input df. Only searches for one component rule per day.
        Returns dataframe of dates with significantly strange events and the corresponding features.
        """

        # reduce dataset to days of interest
        self.reduce_dataset()
        
        # if no data is shared
        if len(self.df) == 0:
            return None

        # initialize
        self.alpha = self.alpha_single
        self.remaining_ftcols = self.ft_col.copy()
        self.get_unique_feature_values()
            
        # get current and reference datasets
        self.get_current_cases()

        # search one component rules
        ftcol, feat, score, pval = self.search_one_component(curr = self.curr_df,
                                                             ref = self.ref_df)

        # save significant results
        if feat is not None:
            return pd.DataFrame({'features': [feat],
                                  'scores': [score],
                                  'dates': [self.today],
                                  'p_value': [pval]})
        else:
            return None
          
    def full_search_n_component(self, n):
        
        """
        Runs full search on each day in input df. Considers multiple features in each rule.
        Each rule can have at most one value from each feature. Returns dataframe of dates 
        with significantly strange events and the corresponding features.
        """

        if n > len(self.ft_col):
            raise ValueError('Error: n must be <= number of feature columns.')

        # reduce dataset to days of interest
        self.reduce_dataset()
        
        # if no data is shared
        if len(self.df) == 0:
            return None

        # initialize
        self.create_results_dataframe()
        self.initialize_components()
        self.get_unique_feature_values()

        keep_score = 1
        keep_pval = 1
        keep_pval_n = None

        self.alpha = self.alpha_single

        # get current and reference datasets
        self.get_current_cases()

        # search one component rules
        ftcol, feat, score, pval = self.search_one_component(curr = self.curr_df,
                                                             ref = self.ref_df)

        # save component
        self.rule_cols.append(ftcol)
        self.rule_vals.append(feat)

        # save scores
        keep_score = score
        keep_pval = pval

        # if one component rule is detected
        if feat is not None:

            # refine datasets
            self.refine_datasets()

            # change alpha value
            self.alpha = self.alpha_n

            # remove significant feature column from future tests
            self.remaining_ftcols.remove(ftcol)

            # iterate through remaining feature columns
            while (len(self.remaining_ftcols) > 0) & (len(self.rule_cols) < n):

                # search one component rules on refined datasets
                ftcol_2, feat_2, score_2, pval_2 = self.search_one_component(curr = self.curr_df_refined,
                                                                             ref = self.ref_df_refined)

                # if no additional significant feature
                if feat_2 is not None:

                    # test significance of adding new component to rule
                    pval_3 = self.significance_of_new_component(ftcol = ftcol_2,
                                                              ftval = feat_2)

                    if pval_3:

                        # store new component
                        self.rule_cols.append(ftcol_2)
                        self.rule_vals.append(feat_2)

                        # store scores
                        keep_score = score_2
                        keep_pval = pval_2
                        keep_pval_n = pval_3

                        # refine datasets further
                        self.refine_datasets()

                        # remove feature column from further consideration
                        self.remaining_ftcols.remove(ftcol_2)

                    else:
                        break

                else:
                    break

            # save results
            self.update_results(keep_score, keep_pval, keep_pval_n)

            return self.results
        
        else:
            return None


## additional helper functions

def get_simulated_and_reference_populations_uniform(df_file, ref_pop_file, fip):
    
    """
    Returns the simulated dataset and the full reference population. The full reference population
    is formatted to be used by WSARE_dynamic_PK - uniform version.
    """
    
    # simulated surveillance data
    df = pd.read_pickle(df_file).astype({'age':int})

    # reference population
    census = pd.read_csv(ref_pop_file)
    pop_ref = census[census.fips == fip].\
              reset_index(drop = True).drop(['fips'], axis=1)

    # create full population reference
    dates = df['date'].unique()
    start_date = pd.to_datetime(dates).min()
    end_date = pd.to_datetime(dates).max()

    a = [pop_ref.age.unique(),
         pop_ref.race.unique(),
         pop_ref.sex.unique(),
         pop_ref.ethnicity.unique(),
         pd.date_range(start_date, end_date).astype(str)]

    full_pop_ref = pd.DataFrame(data = list(itertools.product(*a)),
                               columns = ['age', 'race', 'sex', 'ethnicity', 'date'])
    
    return df, full_pop_ref

def format_popref(df, fip):

    """
    Formats the pop_ref df for WSARE_dynamic_PK.
    """

    # filter by fips code
    df = df[df.fips == fip].\
              reset_index(drop = True).drop(['fips'], axis=1)

    # row level format
    dataset_rowlevel = pd.DataFrame(
        data = np.concatenate(list(map(lambda i: np.tile(df.loc[i,
            ['age','sex', 'race','ethnicity']].values,
            [df.loc[i, ['counts']][0], 1]),
        range(len(df)))),
        axis=0),
        columns = ['age','sex', 'race','ethnicity'])

    return dataset_rowlevel

def get_simulated_and_reference_populations_popref(df_file, pop_ref_file, n_samples=None):
    
    """
    Returns the two files, used for WSARE_dynamic_PK - popref version.
    Since the full dataset is unwieldy for WSARE, the user can specify how many samples
    from the row-level dataset (assumed to be in pop_ref_file) should be randomly sampled
    without replacement.
    """
    
    # simulated surveillance data
    df = pd.read_pickle(df_file).astype({'age':int})

    # reference population
    pop_ref = pd.read_csv(pop_ref_file)

    if n_samples is not None:

        rng = np.random.default_rng()
        random_idx = rng.choice(pop_ref.index, size = n_samples, replace = False)
        pop_ref = pop_ref.iloc[random_idx].reset_index(drop=True)
    
    return df, pop_ref

def policy_dataset_counts(df,
                          ages = False,
                          races = False,
                          sexes = False,
                          ethnicities = False,
                          combined_race_ethnicity = False,
                          months = False):

    """
    Generalizes counts per bin in shared dataset for age, race, sex, and/or ethnicity according to a single policy.
    Returns the averaged aggregated bin values for the policy, where the dataframe's index values matches that of 
    the original for bins with non-zero counts. This function effectively prepares the data for the KL-divergence 
    calculation as in Weiyi's RU-policy frontier paper. The number of counts per bin needs to be in a column
    labeled 'counts'.
    """

    temp = df.copy()
    
    # generalize age, race, sex, and ethnicity values
    if ages:
        temp['age'] = pd.cut(temp['age'],ages, right=False) 
    if races:
        for race_gen in races:
            for key, value in race_gen.items():
                for val in value:
                    temp.loc[temp.race == val, 'race'] = key         
    if sexes:
        temp['sex'] = 'both_sex'
    if ethnicities:
        temp['ethnicity'] = 'both_ethnicity'
    if combined_race_ethnicity: #all hl ethnicity becomes HL race, also removes ethnicity from final df
        temp.loc[temp.ethnicity == 'hl', 'race'] = 'HL'
    if months: # convert date of diagosis to month diagnosis
        temp['date'] = pd.to_datetime(temp['date']).dt.to_period('M').astype(str)

    # group and average by new values
    final = temp.\
                merge(temp.\
                           groupby(['race', 'age', 'sex', 'ethnicity', 'date']).\
                           agg(deid_counts = pd.NamedAgg('counts', 'mean')),
                       left_on = ['race', 'age', 'sex', 'ethnicity', 'date'],
                       right_index = True)
    
    return final



