"""
Code for generating synthetic infectious diseases surveillance data with simulated
subpopulation outbreaks.
"""

import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats.distributions import lognorm
import matplotlib.pyplot as plt




## Helper function to generate log-normal shaped outbreaks

def simulate_lognormal_outbreak(len_ob, peak_val, peak_day, sigma, noise_factor, mu = None):
    
    """
    Returns lognormal-shaped PDF to be used for outbreak simulation. The returned values are the additional
    proportion of daily cases the outbreak group should cover.
    ---------------------
    noise_factor: float, limits magnitude of noise e.g. 10 => max(potential noise) = 10% of peak_val,
                  recomended value: [0, 10]     
    """
    
    # calculate mu
    if mu is None:
        mu = np.log(peak_day) + sigma**2
        #print('mu:', mu)
    
    # calculate parameters for lognormal PDF
    s = sigma
    scale = np.exp(mu)
    
    # days
    days = np.arange(1, len_ob + 1)
    
    # get clean PDF
    pdf = lognorm(s=s, scale=scale).pdf(days)
    
    # add noise
    if noise_factor == 0:

        noisy_pdf = np.abs(pdf)
        
    else:
        
        sd = peak_val * noise_factor / 400
        #print('Noise SD', sd)
        rng = np.random.default_rng()
        noise = rng.normal(loc = 0, scale = sd, size = len_ob)
        noisy_pdf = np.abs(pdf + noise)
    
    # normalize
    max_val = max(noisy_pdf)

    factor = peak_val / max_val

    normalized_pdf = noisy_pdf * factor
    print('Max normalized pdf:',max(normalized_pdf))

    plt.scatter(np.arange(len(normalized_pdf)), normalized_pdf)
    
    return normalized_pdf


## Synthesis class

class Synthesize_outbreak_data:
    
    def __init__(self, counts, gen_census, outbreaks, rng=np.random.default_rng()):
        
        self.counts = counts
        self.dates = counts.columns
        self.census = gen_census
        self.n_bins = len(self.census.columns)
        self.outbreaks = outbreaks
        self.rng = rng
        self.dataset = pd.DataFrame(index = self.dates,
                                    data = np.zeros((len(self.dates), self.n_bins)),
                                    columns = self.census.columns)
        
    def initialize_uninfected_population(self):

        """
        Initializes full population as uninfected.
        """

        self.uninfected = self.census.values.copy()[0]
        
    def update_uninfected_population(self):
        
        """
        Updates uninfected population.
        """
        
        self.uninfected = self.census.values.copy()[0] - self.dataset.sum(axis=0)
        
    def index_population(self, population, indices=None):
        
        """
        Indexes population for downstream sampling.
        ---------------
        population: array of counts per bin
        indices: array of indices for selective indexing
        """
        
        if indices is not None:
             return np.concatenate(
            list(
                map(
                    lambda i: np.repeat(i, population[i]), indices)),
            axis=0)
            
        else:
            return np.concatenate(
                list(
                    map(
                        lambda i: np.repeat(i, population[i]), range(len(population)))),
                axis=0)

        
    def choose_individuals(self, indexed_pop, n_samples):

        """
        Monte Carlo random samples without replacement the infected indviduals from the population.
        ---------------
        indexed_pop: output of self.index_population
        n_samples: number of samples to be taken from self.index_population
        """
        
        return self.rng.choice(indexed_pop,
                               size = int(n_samples),
                               replace = False)

    def random_selection(self, n_samples, population, indices = None):

        """
        Creates the infected population for each time period.
        """
        
        indexed_pop = self.index_population(population = population, indices = indices)
        
        return self.choose_individuals(indexed_pop, n_samples)
        
            
    def add_per_bin(self):

        """
        Adds the number of infected individuals in each demographic bin to the current time
        period's infections. Counts are added to the overall dataset.
        """

        # split shuffled values on sample size
        samples, self.infected = np.split(self.infected, [int(self.n_records)])
        
        # aggregated number of records per bin
        agg_samples = Counter(samples)
        
        # add new infected people per bin
        for i_bin in agg_samples:
            self.dataset.loc[self.day, self.census.columns[i_bin]] += agg_samples[i_bin]
            
    def remove_per_bin(self):
        
        """
        Counts the number of infected individuals in each demographic bin for the current time
        period's infections. Counts are added to the
        """
        
        # aggregated number of records per bin
        agg_samples = Counter(self.removed)
        
        # add new infected people per bin
        for i_bin in agg_samples:
            self.dataset.loc[self.day, self.census.columns[i_bin]] -= agg_samples[i_bin]
                                     
    def fill_zeros(self):

        """
        Fills each cell with zero for a day with zero new cases.
        """

        self.dataset.loc[self.day, :] = [0] * self.n_bins
        
    def get_columns_reference(self):
        
        """
        Defines reference dataframe used in finding the indices of the outbreak bins.
        """
        
        self.col_ref = pd.DataFrame(data = np.stack(self.dataset.columns.str.split('__')),
                            columns = ['sex', 'race', 'ethnicity', 'age']).reset_index().astype({'age':int})
        
    def get_outbreak_bins_indices(self, outbreak_feats):
        
        """
        Defines the indices corresponding to the outbreak bins.
        """
        
        # if age is included in outbreak
        if 'age' in outbreak_feats.keys():
            
            feat_df = pd.DataFrame({'age': outbreak_feats['age']})
            
            for col, val in outbreak_feats.items():
                
                if col != 'age':
                    
                    feat_df.loc[:,col] = val
                    
        # if age is not included in outbreak
        else:
            
            feat_df = pd.DataFrame(data = [0])
            
            for col, val in outbreak_feats.items():

                feat_df[col] = val
    
            feat_df.drop(0, axis=1, inplace=True)
        
        # indices of outbreak bins
        self.outbreak_idx = pd.merge(self.col_ref, feat_df, how='inner')['index'].values
        
        # indices of non-outbreak bins
        self.non_outbreak_idx = np.setdiff1d(np.arange(len(self.dataset.columns)), self.outbreak_idx)
        
        
    def get_simulation_periods(self):
        
        """
        Identifies the periods of indices corresponding to subpopulation outbreaks (defined by self.outbreaks)
        and random infections.
        """
        
        # first outbreak details
        first_outbreak_start = np.where(self.dates == self.outbreaks[0]['start_date'])[0][0]
        first_outbreak_duration = self.outbreaks[0]['duration']
        first_outbreak_end = first_outbreak_start + first_outbreak_duration - 1

        # add first periods

        self.periods = [{'dates': [0, first_outbreak_end],
                         'outbreak':None}]
        self.periods.append({'dates': [first_outbreak_start, first_outbreak_end],
                             'outbreak':0})

        # set next period start date
        next_start = first_outbreak_end + 1

        # add additional periods
        for i in range(1, len(self.outbreaks)):

            outbreak_start = np.where(self.dates == self.outbreaks[i]['start_date'])[0][0]
            outbreak_duration = self.outbreaks[i]['duration']
            outbreak_end = outbreak_start + outbreak_duration - 1

            self.periods.append({'dates': [next_start, outbreak_end],
                                 'outbreak':None})
            self.periods.append({'dates': [outbreak_start, outbreak_end],
                                 'outbreak':i})

            next_start = outbreak_end + 1

        # add last period, if applicable
        if next_start < len(self.dates) - 1:

            self.periods.append({'dates': [next_start, len(self.dates) - 1],
                                 'outbreak':None})

    def run_full_simulation(self):
        
        """
        Runs the full simulation.
        """

        # initialize uninfected population
        self.initialize_uninfected_population()

        # get periods of unweighted and weighted sampling
        self.get_simulation_periods()
        
        # set column reference for outbreak sampling
        self.get_columns_reference()

        # iterate through periods
        for period in self.periods:
            
            outbreak = period['outbreak']
            
            # if baseline sampling
            if outbreak is None:
                print('baseline!')
                print()

                # get range of dates in time period
                outbreak_dates = period['dates']
                dates_in_period = self.dates[outbreak_dates[0]: outbreak_dates[1] + 1]

                # calculate total number of infections in time period
                total_ppl = self.counts[dates_in_period].sum(axis=1).values[0]

                if total_ppl > 0:

                    # sample infections
                    self.infected = self.random_selection(n_samples = total_ppl, population = self.uninfected)

                    # add infections to bins
                    for self.day in dates_in_period:
                        self.n_records = self.counts[self.day].values[0]

                        if self.n_records == 0:
                            self.fill_zeros()
                        else:
                            self.add_per_bin()

                self.update_uninfected_population()
        
            # if simulating outbreak
            else:
                print('outbreak!')
                # get range of dates in outbreak period
                period_dates = period['dates']
                self.outbreak_dates = self.dates[period_dates[0]: period_dates[1] + 1]
                
                # get outbreak/non outbreak bin indices
                print(self.outbreaks[outbreak]['group'])
                self.get_outbreak_bins_indices(self.outbreaks[outbreak]['group'])

                # all cases
                all_cases = self.dataset.loc[self.outbreak_dates, :].sum(axis=1)

                # cases in outbreak bins
                outbreak_bins_cases = self.dataset.loc[self.outbreak_dates,
                                                       self.dataset.columns[self.outbreak_idx]].sum(axis=1)

                # baseline proportion of outbreak_bins
                outbreak_baseline_prop = outbreak_bins_cases/all_cases
                
                print('mean:', np.nanmean(outbreak_baseline_prop))
                print('sd:', np.nanstd(outbreak_baseline_prop))

                # standard deviation of proportion (ignores null values)
                sd = np.nanstd(outbreak_baseline_prop.values)

                # outbreak peak value
                peak_val = min([sd * self.outbreaks[outbreak]['magnitude'],
                                1]) # ensures to never exceed one

                # raise error if not outbreak index cases already in window - 
                # suggests there may not be outbreak index cases in population
                if peak_val == 0: 
                    print('No outbreak cases in population.')
                    print('Setting peak_val to 0.5')
                    peak_val = 0.5
                    #raise ValueError('No outbreak cases in population.')

                print('peak_val:', peak_val)
                print()

                # get lognormal-shaped epicurve to define added outbreak proportions
                outbreak_props = simulate_lognormal_outbreak(len_ob = self.outbreaks[outbreak]['duration'],
                                                             peak_val = peak_val,
                                                             peak_day = self.outbreaks[outbreak]['peak_day'],
                                                             sigma = self.outbreaks[outbreak]['sigma'],
                                                             noise_factor = self.outbreaks[outbreak]['noise_factor'])

                # calculate the number of additional outbreak cases per day
                n_daily_outbreak_cases = np.ceil(outbreak_props * all_cases.values).astype(int)
                
                # iterate through dates of outbreak - necessary to avoid negative counts
                for i in range(len(n_daily_outbreak_cases)):
                    
                    self.day = self.outbreak_dates[i]
                    
                    # count the number of infected cases per bin
                    cases_per_bin = self.dataset.loc[self.day, :].values

                    # available cases to be changed
                    n_available_records = int(cases_per_bin[self.non_outbreak_idx].sum())

                    # define number of records to be flipped
                    self.n_records = max([n_daily_outbreak_cases[i], 1]) # at least one case
                    self.n_records = min([self.n_records,
                                          self.uninfected[self.outbreak_idx].sum()]) # limited to available uninfected population

                    # at most, all non-outbreak bin records can be flipped
                    self.n_records = min([self.n_records, n_available_records])
                    
                    # only if there are cases to remove, and those cases are available
                    if (sum(cases_per_bin) > 0)&(self.n_records > 0):

                        # check
                        if self.n_records > n_available_records:
                            print("Check:", n_records, n_available_records)

                        # only if there are sufficient cases to remove from cases_per_bin in non_outbreak_idx
                        # and sufficient cases to remove from self.uninfected in outbreak_idx
                        if (sum(cases_per_bin[self.non_outbreak_idx])>0) & (sum(self.uninfected[self.outbreak_idx]) > 0):
                    
                            # randomly select who to remove from non outbreak bins during outbreak period
                            self.removed = self.random_selection(n_samples = self.n_records,
                                                                 population = cases_per_bin,
                                                                 indices = self.non_outbreak_idx)

                            # remove cases from dataset
                            self.remove_per_bin()

                            # randomly select additional cases in outbreak bins
                            self.infected = self.random_selection(n_samples = self.n_records,
                                                                  population = self.uninfected,
                                                                  indices = self.outbreak_idx)

                            # add additional outbreak infections to bins
                            self.add_per_bin()
                    
                    # update uninfected population
                    self.update_uninfected_population()
            
    def get_cumulative_counts(self):

        """
        Calculates cumalitive case counts by demographic bin.
        """

        cum_counts = self.dataset.sum().to_frame().reset_index()
        cum_counts.columns = ['bins', 'counts']
        s = cum_counts.bins.str.split(r'__')
        self.cum_counts = pd.concat([
            pd.DataFrame.from_dict(dict(zip(s.index, s.values))).T,
            cum_counts['counts']],
            axis=1)
        self.cum_counts.columns = ['sex', 'race', 'ethnicity', 'age', 'counts']
        
    def to_bin_format(self):
        
        """
        Formats the dataset into counts by bins, with demographic and date columns.
        """
        
        by_bin = self.dataset.melt(ignore_index = False).reset_index()
        by_bin.columns = ['date', 'bins', 'counts']
        s = by_bin.bins.str.split(r'__')
        self.dataset_binned = pd.concat([
            pd.DataFrame.from_dict(dict(zip(s.index, s.values))).T,
            by_bin[['date', 'counts']]],
            axis=1).astype({'counts':'int'})
        self.dataset_binned.columns = ['sex', 'race', 'ethnicity', 'age', 'date', 'counts']
        
    def to_row_format(self):
        
        """
        Formats the dataset to person-level. This function drops empty bins.
        """
        
        try:
            self.dataset_binned.head()
        except:
            self.to_bin_format()
            
        reduced = self.dataset_binned[self.dataset_binned.counts != 0].reset_index(drop=True)
        reduced['date'] = pd.to_datetime(reduced['date'])
        self.dataset_rowlevel = pd.DataFrame(
            data = np.concatenate(list(map(lambda i: np.tile(reduced.loc[i,
                                                                         ['age','sex', 'race',
                                                                          'ethnicity','date']].values,
                                                             [reduced.loc[i, ['counts']][0], 1]),
                                           range(len(reduced)))),
                                  axis=0),
            columns = ['age','sex', 'race','ethnicity','date']).\
        sort_values('date').\
        reset_index(drop=True)




## Format outbreak details

def format_outbreak_details(outbreaks):

	"""
	Function to format the dictionary of outbreak details, that is input into the Synthesize_outbreak_data class,
	into a dataframe. This dataframe can be saved as reference to describe the subpopulation outbreaks in the 
	synthetic data, as well as used in evaluating how well the data enabled WSARE to detect the outbreaks.
	"""

	outbreak_df = pd.DataFrame(columns = ['outbreak_id','start_date', 'duration',
                                      'peak_day', 'race', 'ethnicity', 'sex', 'age_group'])
	row = 0
	for num, outbreak in outbreaks.items():
	    outbreak_df.loc[row, 'start_date'] = outbreak['start_date']
	    outbreak_df.loc[row, 'duration'] = outbreak['duration']
	    outbreak_df.loc[row, 'peak_day'] = outbreak['peak_day']
	    outbreak_df.loc[row, 'outbreak_id'] = num
	    
	    for col, val in outbreak['group'].items():
	        if col =='age':
	            outbreak_df.loc[row, 'age_group'] = pd.cut(val, [0, 10, 20, 30, 40, 50, 60, 70, 80, 120],
	                                                       right=False).astype(str)[0]
	        else:
	            outbreak_df.loc[row, col] = val
	    row += 1

	return outbreak_df

