from dimensions_reduction import *
import pandas as pd
from dimensions_reduction.missing_values import clean_columns_df


def pipeline_single(df, mv = 0, dimredType = '', lag = 0, save = False, df_name = ''):
    suffix = ''
    if mv != 0:
        df = clean_columns_df(df, mv)
        suffix += "_{}".format(str(mv))
    if dimredType == 'PCA':
        # Choose which column to keep
        countries = df['area']
        years = df['year']
        pop = df['SP.POP.TOTL']
        df = df.drop(columns = ['area', 'year'])
        # Fitting PCA
        df = transform_pca(df)
        # Useless ?
        # for i in range(len(names)):
        #     names[i] = "WB." + names[i]
        # df.columns = names
        df.insert(0, 'SP.POP.TOTL', pop)
        df.insert(0, 'year', years)
        df.insert(0, 'area', countries)
        suffix += "_PCA"

    elif dimredType == 'VT':
        threshold = 0.015 * (1 - .8)
        df = variance_threshold(df, threshold)
        suffix += "_Var"

    if lag != 0:
        df['year'] = df['year'] + lag
        suffix += "_lag{}".format(lag)

    if save and df_name != '':
        df_name = '.'.split(dataset)[0]
        df.to_csv(df_name + suffix + '.csv', index = False)

    return(df)


def clean_country_names(df, type, save = False):
    if type == 'FAO':
        countries_dict = {'Bolivia (Plurinational State of)': 'Bolivia',
                          'Venezuela (Bolivarian Republic of)': 'Venezuela'}
        dict = {'area': countries_dict}

        data_FAO_df = df

        countries = data_FAO_df['area']

        for i in range(len(countries)):
            if countries[i] in countries_dict.keys():
                countries[i] = countries_dict[countries[i]]

        data_FAO_df['area'] = countries
        if save == True:
            data_FAO_df.to_csv('../datasets/clean_datasets/FAOSTAT_Replaced_Countries.csv', index = False)

        return(data_FAO_df)

    if type == 'WB':
        countries_dict = {'Bahamas, The': 'Bahamas', 'Egypt, Arab Rep.': 'Egypt',
                          'Iran, Islamic Rep.': 'Iran (Islamic Republic of)', 'Korea, Rep.': 'Republic of Korea',
                          'Kyrgyz Republic': 'Kyrgyzstan', 'Moldova': 'Republic of Moldova',
                          'Slovak Republic': 'Slovakia', 'Venezuela, RB': 'Venezuela'}
        dict = {'area': countries_dict}

        data_WORLDBANK_df = df
        # data_WORLDBANK_df.replace(countries_dict)

        for idx, row in data_WORLDBANK_df.iterrows():
            if row['area'] in countries_dict.keys():
                data_WORLDBANK_df.at[idx, 'area'] = countries_dict[row['area']]

        if save == True:
            data_WORLDBANK_df.to_csv('../clean_datasets/Worldbank_Replaced_Countries.csv', index = False)

        return(data_WORLDBANK_df)


def pipeline_multiple(mv_before = 0, mv_after = 0, dimred_type_before = '', dimred_type_after = '', lag = 0, save = False):
    data_wb = pd.read_csv('../datasets/base_datasets/WORLDBANK.csv')
    data_fao = pd.read_csv('../datasets/base_datasets/FAOSTAT.csv')
    data_mortality = pd.read_csv('../datasets/base_datasets/mortality_clean_aggregate.csv')
    wb_clean = clean_country_names(data_wb, 'WB')
    fao_clean = clean_country_names(data_wb, 'FAO')
    wb_processed = pipeline_single(wb_clean, mv_before, dimred_type_before, lag)
    fao_processed = pipeline_single(fao_clean, mv_before, dimred_type_before, lag)
    wb_mortality = pd.merge(data_mortality, wb_processed, how='inner', on=['area', 'year'])
    df_merged = pd.merge(wb_mortality, fao_processed, how='inner', on=['area', 'year'])
    df_final = pipeline_single(df_merged, mv_after, dimred_type_after, 0)
    return(df_final)


pipeline_multiple(0, 0, '', '')