import pandas as pd
import csv
import collections

# Recipe inputs

data_WORLDBANK_df = pd.read_csv('Worldbank_Replaced_Countries_Var.csv')
mortality_aggregate_df = pd.read_csv('mortality_clean_aggregate.csv')
print(data_WORLDBANK_df.shape)
print(mortality_aggregate_df.shape)

data_output = pd.merge(mortality_aggregate_df, data_WORLDBANK_df, how = 'inner', on = ['area', 'year'])
print(data_output.shape)
data_output.to_csv('Worldbank_Mortality_Var.csv')