import pandas as pd
from preprocessing.dimensions_reduction.missing_values import clean_columns_df
from preprocessing.dimensions_reduction.PCA import *
from preprocessing.dimensions_reduction.variance_threshold import *


def pipeline_single(df, mv = 0, dimredType = '', lag = 0, save = False, df_name = ''):
    suffix = ''
    if mv != 0:
        df = clean_columns_df(df, mv)
        suffix += "_{}".format(str(mv))

    # Choose which column to keep before dimensions reduction
    countries = df['area']
    years = df['year']
    found_pop = False
    found_type = False
    found_sum = False
    if 'sum' in df.columns.values:
        sum = df['sum']
        found_sum = True
        df = df.drop(columns=['sum'])
    if 'SP.POP.TOTL' in df.columns.values:
        pop = df['SP.POP.TOTL']
        found_pop = True
        df = df.drop(columns=['SP.POP.TOTL'])
    elif 'TOTAL_POP' in df.columns.values:
        pop = df['TOTAL_POP']
        found_pop = True
        df = df.drop(columns=['TOTAL_POP'])
    df = df.drop(columns=['area', 'year'])

    if 'area_code' in df.columns.values:
        df = df.drop(columns = ['area_code'])

    if 'type' in df.columns.values:
        type = df['type']
        found_type = True
        df = df.drop(columns = ['type'])

    if dimredType == 'PCA':
        # Fitting PCA
        df = transform_pca(df)
        # Useless ?
        # for i in range(len(names)):
        #     names[i] = "WB." + names[i]
        # df.columns = names
        suffix += "_PCA"

    elif dimredType == 'VT':
        threshold = 0.015 * (1 - .8)
        df = variance_threshold(df, threshold)
        suffix += "_Var"

    if found_pop:
        df.insert(0, 'TOTAL_POP', pop)
    df.insert(0, 'year', years)
    df.insert(0, 'area', countries)

    if found_sum:
        df.insert(0, 'sum', sum)

    if found_type:
        df.insert(0, 'type', type)

    if lag != 0:
        df['year'] = df['year'] + lag
        suffix += "_lag{}".format(lag)

    if save and df_name != '':
        #df_name = '.'.split(df_name)[0]
        df.to_csv('../datasets/intermediate_datasets/' + df_name + suffix + '.csv', index = False)

    return(df)


def clean_country_names(df, type, save = False):
    if type == 'FAO':
        countries_dict = {'Bolivia (Plurinational State of)': 'Bolivia',
                          'Venezuela (Bolivarian Republic of)': 'Venezuela'}
        dict = {'area': countries_dict}

        data_FAO_df = df

        countries = data_FAO_df['area']

        # for i in range(len(countries)):
        #     if countries[i] in countries_dict.keys():
        #         countries[i] = countries_dict[countries[i]]

        for idx, row in data_FAO_df.iterrows():
            if row['area'] in countries_dict.keys():
                data_FAO_df.at[idx, 'area'] = countries_dict[row['area']]

        for country in countries_dict.keys():
            countries.loc[country] = countries_dict[country]

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
            data_WORLDBANK_df.to_csv('../datasets/clean_datasets/Worldbank_Replaced_Countries.csv', index = False)

        return(data_WORLDBANK_df)


def pipeline_multiple(mv_before = 0, mv_after = 0, dimred_type_before = '', dimred_type_after = '', lag = 0, save = False):
    # Reading data
    data_wb = pd.read_csv('../datasets/base_datasets/WORLDBANK.csv')
    data_fao = pd.read_csv('../datasets/base_datasets/FAOSTAT.csv')
    data_mortality = pd.read_csv('../datasets/base_datasets/mortality_clean_aggregate.csv')
    # Cleaning datasets
    wb_clean = clean_country_names(data_wb, 'WB', True)
    fao_clean = clean_country_names(data_fao, 'FAO', True)
    # Processing data before merging
    wb_processed = pipeline_single(wb_clean, mv_before, dimred_type_before, lag)
    fao_processed = pipeline_single(fao_clean, mv_before, dimred_type_before, lag)
    wb_mortality = pd.merge(data_mortality, wb_processed, how='inner', on=['area', 'year'])
    # Merging datasets
    df_merged = pd.merge(wb_mortality, fao_processed, how='inner', on=['area', 'year'])
    # Processing data after merge
    df_final = pipeline_single(df_merged, mv_after, dimred_type_after, 0)
    # Relative mortality
    df_final['relative_mortality'] = df_final['sum'] / df_final['TOTAL_POP']
    df_final = df_final.drop(columns=['sum'])

    if save == True:
        name = "ALL"
        if mv_before != 0:
            name += "_MV{}".format(str(mv_before))
        if dimred_type_before != '':
            name += "_" + dimred_type_before
        if lag != 0:
            name += "_lag{}".format(lag)
        name += "_Merged"
        if mv_after != 0:
            name += "_MV{}".format(str(mv_after))
        if dimred_type_after != '':
            name += "_" + dimred_type_after
        df_final.to_csv('../datasets/final_datasets/' + name + '.csv')
    return(df_final)


def pipeline_multiple_lag(mv_before=0, mv_after=0, dimred_type_before='', dimred_type_after='', lag=0, save=False):
    before_suffix = ''
    if mv_before != 0:
        before_suffix += '_MV{}'.format(str(mv_before))
    if dimred_type_before != '':
        before_suffix += '_{}'.format(str(dimred_type_before))
    # # Reading data
    # data_wb = pd.read_csv('../datasets/intermediate_datasets/WORLDBANK.csv')
    # data_fao = pd.read_csv('../datasets/intermediate_datasets/FAOSTAT.csv')
    data_mortality = pd.read_csv('../datasets/intermediate_datasets/mortality_clean_aggregate.csv')
    # # Cleaning datasets
    # wb_clean = clean_country_names(data_wb, 'WB')
    # fao_clean = clean_country_names(data_fao, 'FAO')
    # # Processing data before merging
    # wb_processed = pipeline_single(wb_clean, mv_before, dimred_type_before, lag)
    # fao_processed = pipeline_single(fao_clean, mv_before, dimred_type_before, lag)
    wb_processed = pd.read_csv('../datasets/intermediate_datasets/WORLDBANK' + before_suffix + '.csv')
    fao_processed = pd.read_csv('../datasets/intermediate_datasets/FAOSTAT' + before_suffix + '.csv')
    wb_mortality = pd.merge(data_mortality, wb_processed, how='inner', on=['area', 'year'])
    # Merging datasets
    df_merged = pd.merge(wb_mortality, fao_processed, how='inner', on=['area', 'year'])
    # Processing data after merge
    df_final = pipeline_single(df_merged, mv_after, dimred_type_after, 0)
    # Relative mortality
    df_final['relative_mortality'] = df_final['sum'] / df_final['TOTAL_POP']
    df_final = df_final.drop(columns=['sum'])

    if save == True:
        name = "ALL"
        if mv_before != 0:
            name += "_MV{}".format(str(mv_before))
        if dimred_type_before != '':
            name += "_" + dimred_type_before
        if lag != 0:
            name += "_lag{}".format(lag)
        name += "_Merged"
        if mv_after != 0:
            name += "_MV{}".format(str(mv_after))
        if dimred_type_after != '':
            name += "_" + dimred_type_after
        df_final.to_csv('../datasets/final_datasets/' + name + '.csv')
    return (df_final)

#pipeline_multiple(50, 0, 'VT', '', 0, True)

# for i in range(5, 25, 2):
#     pipeline_multiple_lag(50, 0, 'VT', '', i, True)
data_wb = pd.read_csv('../datasets/clean_datasets/Worldbank_Replaced_Countries.csv')
pipeline_single(data_wb, 30, 'VT', 0, True, 'Worldbank')
data_fao = pd.read_csv('../datasets/clean_datasets/FAOSTAT_Replaced_Countries.csv')
pipeline_single(data_fao, 30, 'VT', 0, True, 'FAO')
