from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import sklearn
import pandas as pd
from models import remove_outliers
import numpy as np
import math

def predict_test():

    df = pd.read_csv('../datasets/final_datasets/ALL_MV50_PCA_Merged_PCA.csv')
    df = df.dropna(subset=['TOTAL_POP'], axis=0)

    # sélection du type de cancer
    df = df[df.type == 'C16']
    df = df.drop('type', axis=1)
    df = remove_outliers(df)
    # variable à prédire : mortalité relative
    X = df.drop(columns=['area', 'year', 'relative_mortality', 'TOTAL_POP', 'Unnamed: 0'], axis=1)
    Y = df.relative_mortality


    X['area'] = df['area']
    X['year'] = df['year']
    X['TOTAL_POP'] = df['TOTAL_POP']

    print(len(X.columns.values))
    print(Y)
    # split
    X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.33,
                                                                                 random_state=5)

    # variables population totale, année et pays mises de coté
    X_results = X_test[['area', 'year', 'TOTAL_POP']]
    X_values = X_train[['area', 'year', 'TOTAL_POP']]

    X_test = X_test.drop('TOTAL_POP', axis=1)
    X_train = X_train.drop('TOTAL_POP', axis=1)

    X_test = X_test.drop('area', axis=1)
    X_train = X_train.drop('area', axis=1)

    X_test = X_test.drop('year', axis=1)
    X_train = X_train.drop('year', axis=1)

    # rf = RandomForestRegressor(bootstrap= False, criterion= 'mae', max_depth= 40, max_features= 32, min_samples_leaf= 2, min_samples_split= 6)
    rf = KNeighborsRegressor(n_neighbors= 3, weights= 'distance')
    print('fit')
    rf.fit(X_train,Y_train)
    print('normal')
    rf_m = metrics(rf, X_test, Y_test, X_train, Y_train, X_results, X_values)
    # knn_m = metrics(knn, X_test, Y_test, X_train, Y_train, X_results, X_values)
    results = pd.DataFrame(data=rf_m,
                           index=[0])

    results.to_csv('rf_normal.csv', index=False)

    df = pd.read_csv('predictions_20112012.csv')
    df = df.dropna(subset=['TOTAL_POP'], axis=0)

    print('yo')
    # sélection du type de cancer
    # df = df[df.type == 'C16']
    # df = df.drop('type', axis=1)
    df = remove_outliers(df)
    # variable à prédire : mortalité relative
    X = df.drop(columns=['area', 'year', 'sum', 'TOTAL_POP', 'Unnamed: 0'], axis=1)
    # print(X)
    print('yo')
    Y = df.sum

    X['area'] = df['area']
    X['year'] = df['year']
    X['TOTAL_POP'] = df['TOTAL_POP']

    print(len(X.columns.values))
    print(Y)
    # split
    X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.33,
                                                                                 random_state=5)

    # variables population totale, année et pays mises de coté
    X_results = X_test[['area', 'year', 'TOTAL_POP']]
    X_values = X_train[['area', 'year', 'TOTAL_POP']]

    X_test = X_test.drop('TOTAL_POP', axis=1)
    X_train = X_train.drop('TOTAL_POP', axis=1)

    X_test = X_test.drop('area', axis=1)
    X_train = X_train.drop('area', axis=1)

    X_test = X_test.drop('year', axis=1)
    X_train = X_train.drop('year', axis=1)

    rf_m = metrics(rf, X_test, Y_test, X_train, Y_train, X_results, X_values)
    # knn_m = metrics(knn, X_test, Y_test, X_train, Y_train, X_results, X_values)
    results = pd.DataFrame(data=rf_m,
                           index=[0])

    results.to_csv('rf_ts.csv', index=False)



def metrics(model, X_test, Y_test, X_train, Y_train, X_results, X_values):
    dic = {}

    # R2
    R2_train = model.score(X_train, Y_train)
    R2_test = model.score(X_test, Y_test)
    dic['R2_test'] = R2_test
    dic['R2_train'] = R2_train

    print("R2_test %s" % R2_test)
    print("R2_train %s" % R2_train)

    # prédiction de la mortalité en volume à partir de la mortalité relative prédite par le modèle
    Y_predicted = model.predict(X_test)
    X_results['predicted_relative_mortality'] = Y_predicted
    X_results['relative_mortality'] = Y_test
    X_results['predicted_mortality'] = X_results['predicted_relative_mortality'] * X_results['TOTAL_POP']
    X_results['predicted_mortality'] = X_results['predicted_mortality'].round()
    X_results['true_mortality'] = X_results['relative_mortality'] * X_results['TOTAL_POP']
    X_values['true_mortality'] = Y_train * X_values['TOTAL_POP']

    # Mean Square Error
    mse_test = np.mean((X_results['true_mortality'] - X_results['predicted_mortality']) ** 2)
    dic['MSE'] = mse_test
    print("Mean Square Error : %s" % mse_test)

    # Root Mean Square Error
    rmse_test = math.sqrt(mse_test)
    dic['RMSE'] = rmse_test
    print("Root Mean Square Error : %s" % rmse_test)

    # Mean Average Error
    mae_test = np.mean(abs(X_results['true_mortality'] - X_results['predicted_mortality']))
    dic['MAE'] = mae_test
    print("Mean Average Error : %s" % mae_test)

    # Relative Average Error
    mean_mortality_test = np.mean(X_results['true_mortality'])
    rel_mae = mae_test / mean_mortality_test
    dic['MPE'] = rel_mae
    print("Relative Average Error : %s" % rel_mae)

    # Mean Absolute Percentage of Error
    absdiff = abs(X_results['true_mortality'] - X_results['predicted_mortality'])
    print(X_results['true_mortality'])
    max_one_true = np.maximum(np.ones(len(X_results['true_mortality'])), X_results['true_mortality'])
    division = np.divide(absdiff, max_one_true)
    div = list(zip(list(X_results['area']), list(X_results['year']), list(division), list(absdiff), list(max_one_true)))
    div.sort(key=lambda x: x[2])
    div.sort(key=lambda x: (x[0], x[1]))
    div = [('Country', 'year', 'APE', 'AE', 'GT')] + div
    # write_csv('../plots/evolution_per_country', div)
    mape_test = np.mean(division)
    dic['MAPE'] = mape_test
    print("Mean Absolute Percentage of Error : %s" % mape_test)
    # Mean Deviation
    mean = np.mean(X_results['true_mortality'])
    md = np.mean(abs(X_results['true_mortality'] - mean))
    dic['MD'] = md
    print("Mean deviation : %s" % md)

    return dic

predict_test()