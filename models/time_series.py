import pandas as pd
import numpy as np
import rpy2
from rpy2.robjects.packages import importr

# import R's "base" package
base = importr('base')
# import R's "utils" package
utils = importr('utils')
forecast = importr('forecast')
lmtest = importr('lmtest')
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# Get R file

dget = rpy2.robjects.r['dget']
predictor = dget('auto_arima.r')


def add_row_pred(df, dico, country, n_pred):
    # Function to add predicted valued to a dataframe
    names = list(df.columns.values)
    for i in range(n_pred):
        df.loc[len(df)] = 0
        df.loc[len(df) - 1, 'year'] = 2011 + i
        df.loc[len(df) - 1, 'area'] = country

    for key, value in dico.items():
        year = dico[key]['max_year']
        if year == 2010:
            for i in range(len(value)):
                df.loc[len(df) - n_pred + i, key] = value['values'][i]
        elif (2010 - year) < n_pred:
            for i in range(len(value) - (2010 - year)):
                df.loc[len(df) - n_pred + (2010 - year) + i, key] = value['values'][i + (2010 - year)]
        else:
            for i in range(len(value)):
                df.loc[len(df) - n_pred + i, key] = np.nan


def predict(data, p, year_pred, n_pred, name):
    # data = dataframe with features to predict
    # p = depth of data to make time serie prediction
    # year_pred = first year of prediction
    # n-pred = number of year of to predict

    data = data.loc[data['type'] == 'C16']
    data = data.loc[data['year'] < year_pred]
    data = data.drop(columns=['type', 'relative_mortality'])

    countries = list(set(data['area']))
    features = list(data.columns.values)
    # Final dataframe
    final_results = pd.DataFrame(columns=features)

    for country in countries:
        print(country)
        predictions = {}
        data_c = data.loc[data['area'] == country]

        for feature in features:
            print(feature)
            if feature not in ['year', 'area']:
                predictions[feature] = {}
                max_year = max(data_c['year'])
                df_r = pandas2ri.py2ri(data_c[['year', feature]])
                feat_pred = predictor(df_r, p, n_pred)
                feat_pred = list(pandas2ri.ri2py(feat_pred))

                predictions[feature]['values'] = feat_pred
                predictions[feature]['max_year'] = max_year

        add_row_pred(final_results, predictions, country, n_pred)

    final_results.to_csv(name + 'time_serie_prediction2.csv', index=False)
    return (final_results)

data = pd.read_csv("/Users/titou/Desktop/Centrale/Option OSY/Projet/Epidemium/datasets/final_datasets/ALL_MV30_VT_Merged.csv")
data2 = pd.read_csv("/Users/titou/Desktop/Centrale/Option OSY/Projet/Epidemium/datasets/final_datasets/ALL_MV50_VT_Merged.csv")
a = predict(data,15,2011,2,"30")
b = predict(data2,15,2011,2, "50")