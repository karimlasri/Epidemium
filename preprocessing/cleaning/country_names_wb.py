import pandas as pd

countries_dict = {'Bahamas, The': 'Bahamas', 'Egypt, Arab Rep.': 'Egypt', 'Iran, Islamic Rep.':'Iran (Islamic Republic of)', 'Korea, Rep.':'Republic of Korea','Kyrgyz Republic':'Kyrgyzstan', 'Moldova':'Republic of Moldova', 'Slovak Republic':'Slovakia', 'Venezuela, RB':'Venezuela'}
dict = {'area': countries_dict}

data_WORLDBANK_df = pd.read_csv('WORLDBANK.csv')
#data_WORLDBANK_df.replace(countries_dict)

for idx, row in data_WORLDBANK_df.iterrows():
    if row['area'] in countries_dict.keys():
        data_WORLDBANK_df.at[idx,'area'] = countries_dict[row['area']]

data_WORLDBANK_df.to_csv('Worldbank_Replaced_Countries.csv')