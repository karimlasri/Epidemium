import pandas as pd
import platform

platform.architecture()

countries_dict = {'Bolivia (Plurinational State of)': 'Bolivia', 'Venezuela (Bolivarian Republic of)': 'Venezuela'}
dict = {'area': countries_dict}

data_FAO_df = pd.read_csv('FAOSTAT.csv')
#data_WORLDBANK_df.replace(countries_dict)

countries = data_FAO_df['area']

# for idx, row in data_FAO_df.iterrows():
#     if row['area'] in countries_dict.keys():
#         data_FAO_df.at[idx,'area'] = countries_dict[row['area']]
for i in range(len(countries)) :
    if countries[i] in countries_dict.keys():
        countries[i] = countries_dict[countries[i]]

data_FAO_df['area'] = countries

data_FAO_df.to_csv('FAOSTAT_Replaced_Countries.csv')

print(data_FAO_df.area.unique())

# Memory error à balle !