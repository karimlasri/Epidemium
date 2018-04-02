import pandas as pd
from preprocessing.dimensions_reduction.missing_values import clean_columns_df
from preprocessing.dimensions_reduction.PCA import *
from preprocessing.dimensions_reduction.variance_threshold import *


def pipeline_single(df, mv = 0, dimredType = '', lag = 0, save = False, df_name = ''):
    # Whole pipeline for a single dataset
    suffix = ''
    if mv != 0:
        df = clean_columns_df(df, mv)
        suffix += "_MV{}".format(str(mv))

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
        suffix += "_VT"

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
    # Cleans country names for base datasets
    if type == 'FAO':
        countries_dict = {'Bolivia (Plurinational State of)': 'Bolivia',
                          'Venezuela (Bolivarian Republic of)': 'Venezuela'}
        dict = {'area': countries_dict}

        data_FAO_df = df

        countries = data_FAO_df['area']

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


def remove_outliers(df):
    indexes = []
    d = {col_name: df[col_name] for col_name in df.columns.values}
    df = pd.DataFrame(data=d)
    print(df.shape)
    df = df.reset_index(drop=True)
    outliers = [('Brazil', 1977), ('Brazil', 1978), ('Colombia', 1981), ('Haiti', 1981), ('Haiti', 1983), ('Honduras', 1982), ('Honduras', 1983), ('Jamaica', 1968), ('Jamaica', 1969), ('Jamaica', 1970), ('Jamaica', 1971), ('Jamaica', 1975), ('Pakistan', 1993), ('Pakistan', 1994), ('Portugal', 2004), ('Portugal', 2005), ('Puerto Rico', 1979), ('Bolivia', 2002), ('Azerbaijan', 2003), ('Grenada', 1974), ('Grenada', 1975), ('Grenada', 1976), ('Grenada', 1977), ('Guadeloupe', 1971), ('Guadeloupe', 1972), ('Guadeloupe', 1973), ('Guadeloupe', 1976), ('Guadeloupe', 1977), ('Guadeloupe', 1978), ('Guadeloupe', 1979), ('Guadeloupe', 1980), ('San Marino', 2011), ('San Marino', 2012), ('San Marino', 2013), ('San Marino', 2014), ('San Marino', 2015)]
    for i in range(df.shape[0]):
        for outlier in outliers:
            if df.iloc[i]['area'] == outlier[0] and df.iloc[i]['year'] == outlier[1]:
                #print("Found {} {}".format(outlier[0], outlier[1]))
                indexes += [i]
    df.drop(df.index[indexes], inplace = True, axis=0)
    print(df.shape)
    df.to_csv('../datasets/clean_datasets/mortality_clean.csv')
    return df

def clean_mortality_outliers():
    PATH_dataset = '../datasets/base_datasets/mortality_clean_aggregate'
    df = pd.read_csv(PATH_dataset + ".csv")
    df = remove_outliers(df)
    df.to_csv('../datasets/clean_datasets/mortality_clean.csv')

def pipeline_multiple(mv_before = 0, mv_after = 0, dimred_type_before = '', dimred_type_after = '', lag = 0, save = False):
    # The whole pipeline from all base datasets
    # Reading data
    data_wb = pd.read_csv('../datasets/base_datasets/WORLDBANK.csv')
    data_fao = pd.read_csv('../datasets/base_datasets/FAOSTAT.csv')

    # Cleaning datasets
    wb_clean = clean_country_names(data_wb, 'WB', True)
    fao_clean = clean_country_names(data_fao, 'FAO', True)
    clean_mortality_outliers()
    data_mortality = pd.read_csv('../datasets/clean_datasets/mortality_clean.csv')

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


def pipeline_multiple_lag(mv_before=0, mv_after=0, dimred_type_before='', dimred_type_after='', cancer_type='', lag=0, save=False):
    # Multiple pipeline that generates dataset with lag
    before_suffix = ''
    if mv_before != 0:
        before_suffix += '_MV{}'.format(str(mv_before))
    if dimred_type_before != '':
        before_suffix += '_{}'.format(str(dimred_type_before))

    data_mortality = pd.read_csv('../datasets/base_datasets/mortality_clean_aggregate.csv')
    wb_processed = pd.read_csv('../datasets/intermediate_datasets/Worldbank' + before_suffix + '.csv')
    fao_processed = pd.read_csv('../datasets/intermediate_datasets/FAOSTAT' + before_suffix + '.csv')

    if lag != 0:
        wb_processed['year'] = wb_processed['year'] + lag
        fao_processed['year'] = fao_processed['year'] + lag

    # Selecting cancer type
    if len(cancer_type) != 0:
        data_mortality=data_mortality[data_mortality.type == cancer_type]

    # Merging dataframes
    wb_fao = pd.merge(fao_processed, wb_processed, how='inner', on=['area', 'year'])
    df_merged = pd.merge(wb_fao, data_mortality, how='left', on=['area', 'year'])

    # Processing data after merge
    df_final = pipeline_single(df_merged, mv_after, dimred_type_after, 0)
    # Relative mortality
    df_final['relative_mortality'] = df_final['sum'] / df_final['TOTAL_POP']
    df_final = df_final.drop(columns=['sum'])

    # Saving dataframe
    if save == True:
        name = "ALL"
        if mv_before != 0:
            name += "_MV{}".format(str(mv_before))
        if dimred_type_before != '':
            name += "_" + dimred_type_before
        name += "_Merged"
        if mv_after != 0:
            name += "_MV{}".format(str(mv_after))
        if dimred_type_after != '':
            name += "_" + dimred_type_after
        if len(cancer_type) != 0:
            name += "_{}".format(cancer_type)
        if lag != 0:
            name += "_Lag{}".format(lag)
        df_final.to_csv('../datasets/final_datasets/' + name + '.csv')
    return (df_final)

pipeline_multiple_lag(30, 0, 'PCA', 'PCA', cancer_type='C16', lag=0, save=True)
