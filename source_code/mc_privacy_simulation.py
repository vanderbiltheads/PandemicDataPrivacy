"""
Functions for parallel Monte Carlo simulations of COVID cases over time.
"""

# import libraries
import pandas as pd
import numpy as np
import itertools

# generalization hierarchy dictionaries

age_hier = {0:[0,150],
            1:[0, 60, 120],
            2:[0, 30, 60, 90, 120],
            3:[0, 15, 30, 45, 60, 75, 90, 120],
            4:[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 120],
            5:False}

age_name = {0:'*',
            1:'4',
            2:'3',
            3:'2',
            4:'1',
            5:'0'}

race_hier = {0:[{'all':['w', 'b', 'ai_an', 'a', 'nh_pi', 'other', 'mixed']}],
             1:[{'wb':['w','b']}, {'notwb':['ai_an', 'a', 'nh_pi', 'other', 'mixed']}],
             2:[{'other':['ai_an', 'nh_pi', 'other', 'mixed']}],
             3:None}

race_name = {0:'*',
             1:'C',
             2:'B',
             3:'A'}

sex_hier = {0:True,
            1:False}

sex_name = {0:'*',
            1:'s'}

ethnicity_hier = {0:True,
                  1:False}

ethnicity_name = {0:'*',
                  1:'e'}

# helper function for generalizing census data according to policy specifications

def generalize(df, ind_col = "fips", ages = False, races = False, sexes = False, ethnicities = False):
    """
    Generalizes census data for age, race, sex, and/or ethnicity.
    """
    temp = df.copy()
    
    if ages:
        temp['age'] = pd.cut(temp['age'],ages, right=False) 
    if races:
        for race_gen in races:
            for key, value in race_gen.items():
                for val in value:
                    temp.loc[temp.race == val, 'race'] = key         
    if sexes:
        temp['sex'] = 'both_sex'
    if ethnicities == 1:
        temp['ethnicity'] = 'both_ethnicity'
        
    new_temp = temp.groupby([ind_col, 'sex', 'race', 'age', 'ethnicity'])\
                    .agg({'counts':'sum'})\
                    .astype({'counts':int}).reset_index()
    
    new_temp['bins'] = new_temp['sex'] + ',' + \
                        new_temp['race'] + ',' + \
                        new_temp['ethnicity'] + ',' + \
                        new_temp['age'].astype(str)
    
    return new_temp.pivot_table(index = ind_col, columns = 'bins', values = 'counts')

# simulation classes

class privacy_risk_estimation_PK:
    
    """
    Uses Monte Carlo sampling techniques (without replacement) to estimate the longitudinal PK risk
    of a sharing patient-level pandemic data on a consistent basis (e.g., daily or weekly)
    for a user-specified k value. The PK risk is estimated for each time point in a given county,
    when sharing data under a specific data sharing policy (which defines the demographic bins). 
    The PK risk values are calculated on a lagged period of infected individuals.
    
    Input:
    counts = Dataframe of the case counts per time period (e.g. daily counts of new disease cases from
             the JHU COVID-19 surveillance data). Must include the fips code in the index and the columns
             must be date values.
    gen_census = The generalized census, i.e., the output of the generalize function above for the
                 specified fips code.
    fips = The fips code of interest. Must be of the same format of the counts dataframe index column.
    n_sims = The number of simulations to be run in the experiment.
    k = The k value to be used in the PK risk calculation. Default is 10.
    period_size = The size of the lagging period to be used for calculating the PK risk values.
                  Default value is 3.
    
    Output:
    self.PK = Dataframe where each row is a unique simulation and each column is a time period. Each cell
              value corresponds to the proportion of infected individuals who fall into a demographic
              bins of size k or less. The self.PK values are calculated from a lagged period of
              individuals, whose size is specified by period_size. For example, if period_size = 3 and
              the dataset is updated daily, the PK risk value on a given day in a given simulation is 
              the proportion of infected individuals from the day and the previous two days who fall into a 
              demographic bins of size k or less.
    """
    
    def __init__(self, counts, gen_census, fips, n_sims, k=10, period_size=3, rng=np.random.default_rng()):
        
        self.counts = counts.loc[fips,:].values
        self.dates = counts.columns
        self.census = gen_census
        self.n_bins = len(self.census)
        self.PK = pd.DataFrame(columns = self.dates)
        self.n_sims = n_sims
        self.xk = np.arange(self.n_sims)
        self.k = k
        self.period_size = period_size
        self.recent_cases = []
        self.rng = rng
        
    def create_full_population(self):
        
        """
        Creates full population from generalized census counts.
        """
        
        self.full_pop = np.tile(self.census.values, (self.n_sims,1))
        
    def get_infected_population(self):
        
        """
        Creates the infected population for each simulation.
        """
        
        ppl = self.full_pop[0]
        self.indexed_pop = np.concatenate(list(map(lambda i: np.repeat(i, ppl[i]), range(len(ppl)))),axis=0)
        
        # if more than one equivalence class, randomly choose infected from full population
        if len(self.full_pop[0]) > 1:
            self.choose_infected()
        else:
            self.choose_infected(False)
        
    def choose_infected(self, true_shuffle=True):
        
        """
        Monte Carlo random samples without replacement the infected indviduals from the population.
        """
        
        total_ppl = self.counts.sum()
        
        if true_shuffle:
            self.infected = np.stack(list(map(lambda sim: self.rng.choice(self.indexed_pop,
                                                                          size=total_ppl,
                                                                          replace=False),
                                         range(self.n_sims))), axis=0)
        else:
            row = self.indexed_pop[:total_ppl]
            self.infected = np.tile(row, (self.n_sims, 1))
            
        #del self.indexed_pop
    
    def count_per_bin(self):
            
        """
        Counts the number of infected individuals in each demographic bin for the current time
        period's infections.
        """
        
        # empty array for the time period's newest per bin per simulation
        self.new_cases = np.zeros((self.n_sims, self.n_bins))
        
        # split shuffled values on sample size
        samples, self.infected = np.split(self.infected, [self.n_ppl], axis=1)
        
        # add infected people per bin
        for i in samples.T:
            self.new_cases[self.xk,i] += 1
    
    def drop_frame(self):
        
        """
        Removes the oldest set of infections from recent cases.
        """
        
        self.recent_cases.pop(0)
        
    def add_frame(self):
        
        """
        Adds the newest set of infections to recent cases.
        """
        
        self.recent_cases.append(self.new_cases)
    
    def update_recent_cases(self):
        
        """
        Updates the list of recent cases reported within the lagging period.
        """
        
        if len(self.recent_cases) == self.period_size:
            self.drop_frame()
            self.add_frame()
        else:
            self.add_frame()
        self.get_cases_in_period()
        
    def get_cases_in_period(self):
        
        """
        Sums the number of reported cases in each demographic bin with a diagnosis date within
        the lagging period.
        """
        
        self.cases_in_period = sum(self.recent_cases)
    
    def calc_PK_risk(self):
        
        """
        Calculates the PK risk on the cases reported in the lagging period.
        """
        
        n_ppl = self.cases_in_period.sum(axis=1)[0]
        if n_ppl == 0:
            return [0] * self.n_sims
        else:
            risk = np.ndarray((self.n_sims, self.k))
            for i in range(1, self.k+1):
                risk[:,(i-1)] = np.count_nonzero(self.cases_in_period == i, axis=1) * i
            return risk.sum(axis=1)/n_ppl
        
    def run_full_simulation(self):
        """
        Runs the full simulation.
        """
        self.create_full_population()
        self.get_infected_population()
        
        fill_zeros = True
        
        for i in range(len(self.dates)):
            date = self.dates[i]
            self.n_ppl = self.counts[i]
            
            if fill_zeros:
                if (self.n_ppl == 0):
                    self.PK[date] = [0] * self.n_sims
                else:
                    self.count_per_bin()
                    self.update_recent_cases()
                    self.PK[date] = self.calc_PK_risk()
                    fill_zeros = False
                
            else:
                self.count_per_bin()
                self.update_recent_cases()
                self.PK[date] = self.calc_PK_risk()
        
    def get_stats(self, df, percentiles):
        
        """
        Helper function to generate summary statistics on the simulation results.
        """
        
        stats = np.percentile(df, percentiles, axis=0)
        results = pd.DataFrame()
        results['date'] = self.dates
        results['lower'] = stats[0, :]
        results['mean'] = np.mean(df, axis=0).values
        results['upper'] = stats[1, :]

        return results

class privacy_risk_estimation_marketer:
    
    """
    Uses Monte Carlo sampling techniques (without replacement) to estimate the longitudinal marketer
    risk of a sharing patient-level pandemic data on a consistent basis (e.g., daily or weekly). The 
    marketer risk is estimated for each time point in a given county, when sharing data under a 
    specific data sharing policy (which defines the demographic bins). The marketer risk values are 
    calculated on the cumulative dataset over time.
    
    Input:
    counts = Dataframe of the case counts per time period (e.g. daily counts of new disease cases from
             the JHU COVID-19 surveillance data). Must include the fips code in the index and the columns
             must be date values.
    gen_census = The generalized census, i.e., the output of the generalize function above for the
                 specified fips code.
    fips = The fips code of interest. Must be of the same format of the counts dataframe index column.
    n_sims = The number of simulations to be run in the experiment.
    
    Output:
    self.marketer = Dataframe where each row is a unique simulation and each column is a time period.
                    Each cell value corresponds to each row's (simulation's) marketer risk on
                    that time period, calculated on the cumulative dataset through that time period.
    """
    
    def __init__(self, counts, gen_census, fips, n_sims, rng=np.random.default_rng()):
        
        self.counts = counts.loc[fips,:].values
        self.dates = counts.columns
        self.census = gen_census
        self.n_bins = len(self.census)
        self.marketer = pd.DataFrame(columns = self.dates)
        self.n_sims = n_sims
        self.xk = np.arange(self.n_sims)
        self.rng = rng
        self.all_cases = np.zeros((self.n_sims, self.n_bins))
        
    def create_full_population(self):
        
        """
        Creates full population from generalized census counts.
        """
        
        self.full_pop = np.tile(self.census.values, (self.n_sims,1))
        
    def get_infected_population(self):
        
        """
        Creates the infected population for each simulation.
        """
        
        ppl = self.full_pop[0]
        self.indexed_pop = np.concatenate(list(map(lambda i: np.repeat(i, ppl[i]), range(len(ppl)))),axis=0)
        
        # if more than one equivalence class, randomly choose infected from full population
        if len(self.full_pop[0]) > 1:
            self.choose_infected()
        else:
            self.choose_infected(False)
        
    def choose_infected(self, true_shuffle=True):
        
        """
        Monte Carlo random samples without replacement the infected indviduals from the population.
        """
        
        total_ppl = self.counts.sum()
        
        if true_shuffle:
            self.infected = np.stack(list(map(lambda sim: self.rng.choice(self.indexed_pop,
                                                                          size=total_ppl,
                                                                          replace=False),
                                         range(self.n_sims))), axis=0)
        else:
            row = self.indexed_pop[:total_ppl]
            self.infected = np.tile(row, (self.n_sims, 1))
            
        #del self.indexed_pop
    
    def count_per_bin(self):
            
        """
        Counts the number of infected individuals in each demographic bin for the current time
        period's infections.
        """
        
        # split shuffled values on sample size
        samples, self.infected = np.split(self.infected, [self.n_ppl], axis=1)
        
        # add new infected people per bin
        for i in samples.T:
            self.all_cases[self.xk,i] += 1
        
    def calc_marketer_risk(self):
        
        """
        Calculates the marketer risk on the cumulative simulated disease case dataset.
        """
        
        with np.errstate(divide='ignore', invalid='ignore'):
            group_ratios = np.nan_to_num(self.all_cases / self.full_pop)
        return group_ratios.sum(axis=1)/self.all_cases.sum(axis = 1)
        
    def run_full_simulation(self):
        
        """
        Runs the full simulation.
        """
        
        self.create_full_population()
        self.get_infected_population()
        
        fill_zeros = True
        
        for i in range(len(self.dates)):
            date = self.dates[i]
            self.n_ppl = self.counts[i]
            
            if fill_zeros:
                if (self.n_ppl == 0):
                    self.marketer[date] = [0] * self.n_sims
                else:
                    self.count_per_bin()
                    self.marketer[date] = self.calc_marketer_risk()
                    fill_zeros = False
                
            else:
                self.count_per_bin()
                self.marketer[date] = self.calc_marketer_risk()
        
    def get_stats(self, df, percentiles):
        
        """
        Helper function to generate summary statistics on the simulation results.
        """
        
        stats = np.percentile(df, percentiles, axis=0)
        results = pd.DataFrame()
        results['date'] = self.dates
        results['lower'] = stats[0, :]
        results['mean'] = np.mean(df, axis=0).values
        results['upper'] = stats[1, :]

        return results

class PK_fair_race:
    
    """
    Uses Monte Carlo sampling techniques (without replacement) to estimate the longitudinal PK risk
    of a sharing patient-level pandemic data on a consistent basis (e.g., daily or weekly)
    for a user-specified k value. The PK risk is estimated for each time point in a given county,
    when sharing data under a specific data sharing policy (which defines the demographic bins). 
    The PK risk values are calculated on a lagged period of infected individuals.
    
    Input:
    counts = Dataframe of the case counts per time period (e.g. daily counts of new disease cases from
             the JHU COVID-19 surveillance data). Must include the fips code in the index and the columns
             must be date values.
    gen_census = The generalized census, i.e., the output of the generalize function above for the
                 specified fips code.
    fips = The fips code of interest. Must be of the same format of the counts dataframe index column.
    n_sims = The number of simulations to be run in the experiment.
    k = The k value to be used in the PK risk calculation. Default is 10.
    period_size = The size of the lagging period to be used for calculating the PK risk values.
                  Default value is 3.
    
    Output:
    self.*race*_PK_prop = Dataframe where each row is a unique simulation and each column is a time period.
                       Each cell value corresponds to the proportion of the total PK risk in the lagging
                       period the corresponding racial group bears. There is a dataframe for each racial
                       group present in the US census data.
    """
    
    def __init__(self, counts, gen_census, fips, n_sims, k=10, period_size=3, rng=np.random.default_rng(),
                 black = [], asian = [], alaska = [], pacific = [], mixed = [], other = [], white = []):
        
        self.counts = counts.loc[fips,:].values
        self.dates = counts.columns
        self.census = gen_census
        self.n_bins = len(self.census)
        self.n_sims = n_sims
        self.xk = np.arange(self.n_sims)
        self.k = k
        self.period_size = period_size
        self.recent_cases = []
        self.rng = rng
        
        # all races
        self.PK = pd.DataFrame(columns = self.dates)
        
        # race-specific
        if len(black) > 0:
            self.black = black
            self.black_PK_prop = pd.DataFrame(columns = self.dates)
            self.black_include = True
        else:
            self.black_include = False
        if len(asian) > 0:
            self.asian = asian
            self.asian_PK_prop = pd.DataFrame(columns = self.dates)
            self.asian_include = True
        else:
            self.asian_include = False
        if len(alaska) > 0:
            self.alaska = alaska
            self.alaska_PK_prop = pd.DataFrame(columns = self.dates)
            self.alaska_include = True
        else:
            self.alaska_include = False
        if len(pacific) > 0:
            self.pacific = pacific
            self.pacific_PK_prop = pd.DataFrame(columns = self.dates)
            self.pacific_include = True
        else:
            self.pacific_include = False
        if len(mixed) > 0:
            self.mixed = mixed
            self.mixed_PK_prop = pd.DataFrame(columns = self.dates)
            self.mixed_include = True
        else:
            self.mixed_include = False
        if len(other) > 0:
            self.other = other
            self.other_PK_prop = pd.DataFrame(columns = self.dates)
            self.other_include = True
        else:
            self.other_include = False
        if len(white) > 0:
            self.white = white
            self.white_PK_prop = pd.DataFrame(columns = self.dates)
            self.white_include = True
        else:
            self.white_include = False
        
    def create_full_population(self):
        
        """
        Creates full population from generalized census counts.
        """
        
        self.full_pop = np.tile(self.census.values, (self.n_sims,1))
        
    def get_infected_population(self):
        
        """
        Creates the infected population for each simulation.
        """
        
        ppl = self.full_pop[0]
        self.indexed_pop = np.concatenate(list(map(lambda i: np.repeat(i, ppl[i]), range(len(ppl)))),axis=0)
        
        # if more than one equivalence class, randomly choose infected from full population
        if len(self.full_pop[0]) > 1:
            self.choose_infected()
        else:
            self.choose_infected(False)
        
    def choose_infected(self, true_shuffle=True):
        
        """
        Monte Carlo random samples without replacement the infected indviduals from the population.
        """
        
        total_ppl = self.counts.sum()
        
        if true_shuffle:
            self.infected = np.stack(list(map(lambda sim: self.rng.choice(self.indexed_pop,
                                                                          size=total_ppl,
                                                                          replace=False),
                                         range(self.n_sims))), axis=0)
        else:
            row = self.indexed_pop[:total_ppl]
            self.infected = np.tile(row, (self.n_sims, 1))
            
        #del self.indexed_pop
    
    def count_per_bin(self):
            
        """
        Counts the number of infected individuals in each demographic bin for the current time
        period's infections.
        """
        
        # empty array for the time period's newest per bin per simulation
        self.new_cases = np.zeros((self.n_sims, self.n_bins))
        
        # split shuffled values on sample size
        samples, self.infected = np.split(self.infected, [self.n_ppl], axis=1)
        
        # add infected people per bin
        for i in samples.T:
            self.new_cases[self.xk,i] += 1
    
    def drop_frame(self):
        
        """
        Removes the oldest set of infections from recent cases.
        """
        
        self.recent_cases.pop(0)
        
    def add_frame(self):
        
        """
        Adds the newest set of infections to recent cases.
        """
        
        self.recent_cases.append(self.new_cases)
    
    def update_recent_cases(self):
        
        """
        Updates the list of recent cases reported within the lagging period.
        """
        
        if len(self.recent_cases) == self.period_size:
            self.drop_frame()
            self.add_frame()
        else:
            self.add_frame()
        self.get_cases_in_period()
        
    def get_cases_in_period(self):
        
        """
        Sums the number of reported cases in each demographic bin with a diagnosis date within
        the lagging period.
        """
        
        self.cases_in_period = sum(self.recent_cases)
    
    def calc_PK_risk(self):
        
        """
        Calculates the PK risk on the cases reported in the lagging period.
        """
        
        n_ppl = self.cases_in_period.sum(axis=1)[0]
        if n_ppl == 0:
            return [0] * self.n_sims
        else:
            risk = np.ndarray((self.n_sims, self.k))
            for i in range(1, self.k+1):
                risk[:,(i-1)] = np.count_nonzero(self.cases_in_period == i, axis=1) * i
            return risk.sum(axis=1)/n_ppl
        
    def calc_proportion_PK(self, field):
        
        """
        Calculates the proportion of the PK risk corresponding to a given race.
        """
        
        n_ppl = self.cases_in_period.sum(axis=1)[0]
        if n_ppl == 0:
            return [0] * self.n_sims
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                field_k = np.ndarray((self.n_sims, self.k))
                all_k = np.ndarray((self.n_sims, self.k))
                for i in range(1, self.k+1):
                    field_k[:,(i-1)] = np.count_nonzero(self.cases_in_period[:,field] == i, axis=1) * i
                    all_k[:,(i-1)] = np.count_nonzero(self.cases_in_period == i, axis=1) * i
                return field_k.sum(axis=1)/all_k.sum(axis=1)
        
    def run_full_simulation(self):
        
        """
        Runs the full simulation.
        """
        
        self.create_full_population()
        self.get_infected_population()
        
        fill_zeros = True
        
        for i in range(len(self.dates)):
            date = self.dates[i]
            self.n_ppl = self.counts[i]
            
            if fill_zeros:
                if (self.n_ppl == 0):
                    self.PK[date] = [0] * self.n_sims
                    if self.black_include:
                        self.black_PK_prop[date] = [0] * self.n_sims
                    if self.asian_include:
                        self.asian_PK_prop[date] = [0] * self.n_sims
                    if self.alaska_include:
                        self.alaska_PK_prop[date] = [0] * self.n_sims
                    if self.pacific_include:
                        self.pacific_PK_prop[date] = [0] * self.n_sims
                    if self.mixed_include:
                        self.mixed_PK_prop[date] = [0] * self.n_sims
                    if self.other_include:
                        self.other_PK_prop[date] = [0] * self.n_sims
                    if self.white_include:
                        self.white_PK_prop[date] = [0] * self.n_sims
                else:
                    self.count_per_bin()
                    self.update_recent_cases()
                    self.PK[date] = self.calc_PK_risk()
                    if self.black_include:
                        self.black_PK_prop[date] = self.calc_proportion_PK(self.black)
                    if self.asian_include:
                        self.asian_PK_prop[date] = self.calc_proportion_PK(self.asian)
                    if self.alaska_include:
                        self.alaska_PK_prop[date] = self.calc_proportion_PK(self.alaska)
                    if self.pacific_include:
                        self.pacific_PK_prop[date] = self.calc_proportion_PK(self.pacific)
                    if self.mixed_include:
                        self.mixed_PK_prop[date] = self.calc_proportion_PK(self.mixed)
                    if self.other_include:
                        self.other_PK_prop[date] = self.calc_proportion_PK(self.other)
                    if self.white_include:
                        self.white_PK_prop[date] = self.calc_proportion_PK(self.white)
                    fill_zeros = False
                
            else:
                self.count_per_bin()
                self.update_recent_cases()
                self.PK[date] = self.calc_PK_risk()
                if self.black_include:
                    self.black_PK_prop[date] = self.calc_proportion_PK(self.black)
                if self.asian_include:
                    self.asian_PK_prop[date] = self.calc_proportion_PK(self.asian)
                if self.alaska_include:
                    self.alaska_PK_prop[date] = self.calc_proportion_PK(self.alaska)
                if self.pacific_include:
                    self.pacific_PK_prop[date] = self.calc_proportion_PK(self.pacific)
                if self.mixed_include:
                    self.mixed_PK_prop[date] = self.calc_proportion_PK(self.mixed)
                if self.other_include:
                    self.other_PK_prop[date] = self.calc_proportion_PK(self.other)
                if self.white_include:
                    self.white_PK_prop[date] = self.calc_proportion_PK(self.white)
        
    def get_stats(self, df, percentiles):
        
        """
        Helper function to generate summary statistics on the simulation results.
        """
        
        stats = np.percentile(df, percentiles, axis=0)
        results = pd.DataFrame()
        results['date'] = self.dates
        results['lower'] = stats[0, :]
        results['mean'] = np.mean(df, axis=0).values
        results['upper'] = stats[1, :]

        return results

class dynamic_policy_search_PK:
    
    """
    Searches the policy space for those that meet the PK risk threshold for each of the counties in
    a total population range.
    
    Input:
    age_hier = Dictionary of age generalization hierarchy, where keys are the numerical levels of the hierarcy
               and the values are the parameters to be passed to the generalization helper function. Key 0
               must correspond to the most generalized level in the hierarchy.
    age_name = Dictionary of the name convention for each level of the age generalization hierarchy. The keys
               should match those of the age_hierarchy, where the values are the named value.
    race_hier = Dictionary of race generalization hierarchy.
    race_name = Dictionary of race generalization names.
    sex_hier = Dictionary of sex generalization hierarchy.
    sex_name = Dictionary of sex generalization names.
    ethnicity_hier = Dictionary of ethnicity generalization hierarchy.
    ethnicity_name = Dictionary of ethnicity generalization names.
    census = Dataframe of the census tract information for each county. Columns include fips code, race,
             age, sex, counts, and ethnicity.
    pop_lower_bound = Integer defining the lower bound of the county population range. The range defines
                      which counties are used in generating the privcacy risk estimates.
    pop_upper_bound = Integer defining the upper bound of the county population range.
    threshold = PK risk threshold.
    percent = Percentile used to compare the PK risk estimates to the PK risk threshold. For 
              example, if the upper bound of the 95% quantile range is used for the comparison, percent
              should be 97.5.
    num_simulations = Integer defining the number of simulations run in each county's PK risk estimates.
    caseloads = List of monotonically increasing numbers, defining the case record thresholds at which
                each policy is evaluted. For the PK risk, these numbers represent the total number of
                case records in the dataset.
    k = k value used in PK risk calculation.
    repeat = Boolean value. If True, the search will consider all policies for each case count threshold.
             If False, the search will remove policies that previously met the privacy risk threshold at 
             lower case counts when testing higher case counts.
    
                
    Output:
    results = Dictionary of the policy search results. The keys are the integers from caseloads. The dictionary
              values are the named policies that meet the PK risk threshold for all counties (with a total
              population in the defined range) when the total number of disease case records is at least the value
              of the corresponding key.
    """

    def __init__(self, age_hier, age_name, race_hier, race_name, sex_hier, sex_name, ethnicity_hier,
                 ethnicity_name, census, pop_lower_bound, pop_upper_bound, threshold, percent, num_simulations,
                 caseloads, k = 10, repeat = False):

        self.age_hier = age_hier
        self.age_name = age_name
        self.race_hier = race_hier
        self.race_name = race_name
        self.sex_hier = sex_hier
        self.sex_name = sex_name
        self.ethnicity_hier = ethnicity_hier
        self.ethnicity_name = ethnicity_name
        self.census = census
        self.county_pop = census.groupby('fips').agg({'counts':'sum'}).sort_index()
        self.lb = pop_lower_bound
        self.ub = pop_upper_bound
        self.threshold = threshold
        self.percent = percent
        self.num_sims = num_simulations
        self.num_cases = caseloads
        self.k = k
        self.repeat = repeat
        
    def list_all_policies(self):
        """
        Generate list of tuples including all unique policy generalization combinations
        given the age, race, sex, and ethnicity hierarchies.
        """
        ages = list(self.age_hier.keys())
        races = list(self.race_hier.keys())
        sexes = list(self.sex_hier.keys())
        eths = list(test.ethnicity_hier.keys())

        all_combinations = list(itertools.product(*[ages,races,sexes,eths]))
        combos = pd.DataFrame({'scale':np.array(all_combinations).sum(axis=1)})
        self.all_policies = [all_combinations[i] for i in combos.sort_values('scale').index.values]
        
    def policy_parameters(self, age_idx, race_idx, sex_idx, eth_idx):
        """
        Extract policy name and parameters from hierarchies.
        """
        self.name = self.age_name[age_idx] + \
                    self.race_name[race_idx] + \
                    self.sex_name[sex_idx] + \
                    self.ethnicity_name[eth_idx]
        self.params = [self.age_hier[age_idx],
                       self.race_hier[race_idx],
                       self.sex_hier[sex_idx],
                       self.ethnicity_hier[eth_idx]]
        
    def run_search(self):
        """
        Execute the policy search.
        """
        self.list_all_policies()
        self.results = {}
        
        for num in self.num_cases:

            policies = self.all_policies.copy()
            acceptable_policies = []

            # fix caseload value
            fixed_df = pd.DataFrame({'01-01-01':np.repeat(num, len(self.county_pop.index))},
                                    index = self.county_pop.index)

            # find fips codes for counties that meet total population range
            select = (self.county_pop > num) & (self.county_pop > self.lb) & (self.county_pop < self.ub)
            fips = np.array(self.county_pop.index[select.counts.values])

            while len(policies) > 0:

                passed = True

                # choose first policy
                levels = policies[0]

                # generalize per policy
                self.policy_parameters(age_idx = levels[0],
                                       race_idx = levels[1],
                                       sex_idx = levels[2],
                                       eth_idx = levels[3])

                generalized_census = generalize(self.census[self.census.fips.isin(fips)],
                                                ages = self.params[0],
                                                races = self.params[1],
                                                sexes = self.params[2],
                                                ethnicities = self.params[3])

                # test policy for each county
                for fip in fips:
                    test = privacy_risk_estimation_PK(counts = fixed_df.loc[fip,:].to_frame().transpose(),
                                                      gen_census = generalized_census.loc[fip,:],
                                                      fips = fip,
                                                      n_sims = self.num_sims,
                                                      k = self.k,
                                                      period_size = 1) # period size is set to 1 as results are
                                                                       # period size agnostic
                    test.run_full_simulation()
                    risk = np.percentile(test.PK, self.percent)
                    if risk > self.threshold:
                        passed = False
                        break
                        
                # if the policy meets the threshold for each county, mark the policy as acceptable and
                # remove from consideration for larger caseloads. Otherwise, remove all parent policies
                # from consideration for the current caseload.
                
                if passed:
                    acceptable_policies.append(self.name)
                    policies = policies[1:]
                    
                    # remove acceptable policies from consideration for subsequent, larger case count
                    # thresholds
                    
                    if not self.repeat: 
                        self.all_policies.remove((int(levels[0]),
                                             int(levels[1]),
                                             int(levels[2]),
                                             int(levels[3])))
                else:
                    keep = []
                    for x, y in enumerate(policies):
                        if not (y[0] >= levels[0]) & (y[1] >= levels[1]) & \
                                (y[2] >= levels[2]) & (y[3] >= levels[3]):
                            keep.append(policies[x])
                    policies = keep.copy()

            # store results
            self.results[num] = acceptable_policies


class dynamic_policy_search_marketer:
    
    """
    Searches the policy space for those that meet the marketer risk threshold for each of the counties in
    a total population range.
    
    Input:
    age_hier = Dictionary of age generalization hierarchy, where keys are the numerical levels of the hierarcy
               and the values are the parameters to be passed to the generalization helper function. Key 0
               must correspond to the most generalized level in the hierarchy.
    age_name = Dictionary of the name convention for each level of the age generalization hierarchy. The keys
               should match those of the age_hierarchy, where the values are the named value.
    race_hier = Dictionary of race generalization hierarchy.
    race_name = Dictionary of race generalization names.
    sex_hier = Dictionary of sex generalization hierarchy.
    sex_name = Dictionary of sex generalization names.
    ethnicity_hier = Dictionary of ethnicity generalization hierarchy.
    ethnicity_name = Dictionary of ethnicity generalization names.
    census = Dataframe of the census tract information for each county. Columns include fips code, race,
             age, sex, counts, and ethnicity.
    pop_lower_bound = Integer defining the lower bound of the county population range. The range defines
                      which counties are used in generating the privcacy risk estimates.
    pop_upper_bound = Integer defining the upper bound of the county population range.
    threshold = Marketer risk threshold.
    percent = Percentile used to compare the marketer risk estimates to the marketer risk threshold. For 
              example, if the upper bound of the 95% quantile range is used for the comparison, percent
              should be 97.5.
    num_simulations = Integer defining the number of simulations run in each county's marketer risk estimates.
    caseloads = List of monotonically increasing numbers, defining the case record thresholds at which
                each policy is evaluted. For the marketer risk, these numbers represent the total number of
                case records in the dataset.
                
    Output:
    results = Dictionary of the policy search results. The keys are the integers from caseloads. The dictionary
              values are the named policies that meet the marketer risk threshold for all counties (with a total
              population in the defined range) when the total number of disease case records is at least the value
              of the corresponding key.
    """

    def __init__(self, age_hier, age_name, race_hier, race_name, sex_hier, sex_name, ethnicity_hier,
                 ethnicity_name, census, pop_lower_bound, pop_upper_bound, threshold, percent, num_simulations,
                 caseloads, repeat = False):

        self.age_hier = age_hier
        self.age_name = age_name
        self.race_hier = race_hier
        self.race_name = race_name
        self.sex_hier = sex_hier
        self.sex_name = sex_name
        self.ethnicity_hier = ethnicity_hier
        self.ethnicity_name = ethnicity_name
        self.census = census
        self.county_pop = census.groupby('fips').agg({'counts':'sum'}).sort_index()
        self.lb = pop_lower_bound
        self.ub = pop_upper_bound
        self.threshold = threshold
        self.percent = percent
        self.num_sims = num_simulations
        self.num_cases = caseloads
        self.repeat = repeat
        
    def list_all_policies(self):
        """
        Generate list of tuples including all unique policy generalization combinations
        given the age, race, sex, and ethnicity hierarchies.
        """
        ages = list(self.age_hier.keys())
        races = list(self.race_hier.keys())
        sexes = list(self.sex_hier.keys())
        eths = list(test.ethnicity_hier.keys())

        all_combinations = list(itertools.product(*[ages,races,sexes,eths]))
        combos = pd.DataFrame({'scale':np.array(all_combinations).sum(axis=1)})
        self.all_policies = [all_combinations[i] for i in combos.sort_values('scale').index.values]
        
    def policy_parameters(self, age_idx, race_idx, sex_idx, eth_idx):
        """
        Extract policy name and parameters from hierarchies.
        """
        self.name = self.age_name[age_idx] + \
                    self.race_name[race_idx] + \
                    self.sex_name[sex_idx] + \
                    self.ethnicity_name[eth_idx]
        self.params = [self.age_hier[age_idx],
                       self.race_hier[race_idx],
                       self.sex_hier[sex_idx],
                       self.ethnicity_hier[eth_idx]]
        
    def run_search(self):
        """
        Execute the policy search.
        """
        self.list_all_policies()
        self.results = {}
        
        for num in self.num_cases:

            policies = self.all_policies.copy()
            acceptable_policies = []

            # fix caseload value
            fixed_df = pd.DataFrame({'01-01-01':np.repeat(num, len(self.county_pop.index))},
                                    index = self.county_pop.index)

            # find fips codes for counties that meet total population range
            select = (self.county_pop > num) & (self.county_pop > self.lb) & (self.county_pop < self.ub)
            fips = np.array(self.county_pop.index[select.counts.values])

            while len(policies) > 0:

                passed = True

                # choose first policy
                levels = policies[0]

                # generalize per policy
                self.policy_parameters(age_idx = levels[0],
                                       race_idx = levels[1],
                                       sex_idx = levels[2],
                                       eth_idx = levels[3])

                generalized_census = generalize(self.census[self.census.fips.isin(fips)],
                                                ages = self.params[0],
                                                races = self.params[1],
                                                sexes = self.params[2],
                                                ethnicities = self.params[3])

                # test policy for each county
                for fip in fips:
                    test = privacy_risk_estimation_marketer(counts = fixed_df.loc[fip,:].to_frame().transpose(),
                                               gen_census = generalized_census.loc[fip,:],
                                               fips = fip,
                                               n_sims = self.num_sims)
                    test.run_full_simulation()
                    risk = np.percentile(test.marketer, self.percent)
                    if risk > self.threshold:
                        passed = False
                        break
                # if the policy meets the threshold for each county, mark the policy as acceptable and
                # remove from consideration for larger caseloads. Otherwise, remove all parent policies
                # from consideration for the current caseload.
                if passed:
                    acceptable_policies.append(self.name)
                    policies = policies[1:]
                    if not self.repeat:
                        self.all_policies.remove((int(levels[0]),
                                             int(levels[1]),
                                             int(levels[2]),
                                             int(levels[3])))
                else:
                    keep = []
                    for x, y in enumerate(policies):
                        if not (y[0] >= levels[0]) & (y[1] >= levels[1]) & \
                                (y[2] >= levels[2]) & (y[3] >= levels[3]):
                            keep.append(policies[x])
                    policies = keep.copy()

            # store results
            self.results[num] = acceptable_policies