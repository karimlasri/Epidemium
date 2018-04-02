import pandas as pd, numpy as np
import os
import math
import sklearn
import sklearn.cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
import csv
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV


def predict_mortality(name, model_name, cancer_type, test_size, developing_countries=False, lag=0):
    # Whole pipeline to predict mortality given the model, and name of dataset to use

    PATH_datasets = '../datasets/final_datasets/'
    df = pd.read_csv(os.path.join(PATH_datasets,name + ".csv"))

    df = df.dropna(subset = ['TOTAL_POP'], axis = 0)

    print(df.info())

    if developing_countries:
        countries_df = pd.read_csv('developping_countries.csv')
        countries = countries_df['area']
        df = df[df.area.isin(countries)]

    print(df.info())

    # sélection du type de cancer
    df=df[df.type == cancer_type]
    df=df.drop('type', axis=1)
    # Suppression des valeurs absurdes
    df = remove_outliers(df)
    # variable à prédire : mortalité relative
    if lag != 0:
        X_lag, X, Y = lag_X_Y(df)
    else:
        X=df.drop(columns = ['area', 'year', 'relative_mortality', 'TOTAL_POP', 'Unnamed: 0'], axis=1)
        Y=df.relative_mortality



    #standardisation des variables d'entrée pour les modèles linéaires
    if model_name in ["linear_regression", "ridge_regression", "lasso_regression"]:
        labels = X.columns
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = pd.DataFrame(scaler.fit_transform(X), columns=labels)

    X['area'] = df['area']
    X['year'] = df['year']
    X['TOTAL_POP'] = df['TOTAL_POP']

    #split
    X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=test_size, random_state = 5)


    #variables population totale, année et pays mises de coté
    X_results=X_test[['area', 'year', 'TOTAL_POP']]
    X_values=X_train[['area', 'year', 'TOTAL_POP']]

    X_test=X_test.drop('TOTAL_POP', axis=1)
    X_train=X_train.drop('TOTAL_POP', axis=1)

    X_test=X_test.drop('area', axis=1)
    X_train=X_train.drop('area', axis=1)

    X_test=X_test.drop('year', axis=1)
    X_train=X_train.drop('year', axis=1)


    if model_name =="linear_regression":
        #pas d'hyperparamètre à déterminer
        #apprentissage du modèle
        model = linear_model.LinearRegression()
        model.fit(X_train, Y_train)
        print("Coefs : {}".format(list(zip(X_train.columns.values, model.coef_))))

    elif model_name == "ridge_regression":

        alphas = []
        train_scores = []
        scores = []
        # alphas in [0.01, 0.1[
        best_alpha = 0.01
        best_score = 0
        for i in range(10):
            alpha = 0.01 * (i)
            alphas.append(alpha)
            reg = linear_model.Ridge(alpha=alpha)
            reg.fit(X_train, Y_train)
            score = cross_val_score(reg, X_train, Y_train, cv=5).mean()
            if score > best_score:
                best_alpha = alpha
                best_score = score
            train_scores += [[alpha, score]]
        # alphas in [0.1, 1[
        for i in range(5):
            alpha = 0.1 * (2 * i)
            alphas.append(alpha)
            reg = linear_model.Ridge(alpha=alpha)
            reg.fit(X_train, Y_train)
            score = cross_val_score(reg, X_train, Y_train, cv=5).mean()
            if score > best_score:
                best_alpha = alpha
                best_score = score
            train_scores += [[alpha, score]]
        # alphas in [1, 19]
        for i in range(10):
            alpha = (2 * i + 1)
            alphas.append(alpha)
            reg = linear_model.Ridge(alpha=alpha)
            reg.fit(X_train, Y_train)
            score = cross_val_score(reg, X_train, Y_train, cv=5).mean()
            if score > best_score:
                best_alpha = alpha
                best_score = score
            train_scores += [[alpha, score]]

        with open('ridge.csv', 'wt') as ridge_csv:
            writer = csv.writer(ridge_csv, delimiter=';')
            writer.writerow(alphas)
            writer.writerow(train_scores)
            writer.writerow(scores)

        print(best_alpha)
        model = linear_model.Ridge(alpha = best_alpha)
        model.fit(X_train, Y_train)


    elif model_name == "lasso_regression":

        alphas = []
        train_scores = []
        scores = []
        non_zero_number = []
        non_zero_values = []
        non_zero_coefs = []
        best_alpha = 0.01
        best_score = 0
        for i in range(10):
            alpha = 0.001 * (i + 1)
            alphas.append(alpha)

            reg = linear_model.Lasso(alpha=alpha)
            reg.fit(X_train, Y_train)
            train_scores.append(reg.score(X_train, Y_train))
            print("Score with Lasso with train, alpha = {} : {}".format(alpha, reg.score(X_train, Y_train)))
            print("Score with Lasso, alpha = {} : {}".format(alpha, reg.score(X_test, Y_test)))
            scores.append(reg.score(X_test, Y_test))
            count = 0
            non_zero_list = []
            non_zero_coefs_list = []
            for i, val in enumerate(reg.coef_):
                if val != 0:
                    count += 1
                    non_zero_list.append(X_train.columns.values[i])
                    non_zero_coefs_list.append(str(reg.coef_[i]))
            non_zero_number.append(count)
            non_zero_coefs.append(non_zero_coefs_list)
            non_zero_values.append(non_zero_list)

        for i in range(10):
            alpha = 0.01 * (i + 1)
            alphas.append(alpha)

            reg = linear_model.Lasso(alpha=alpha)
            reg.fit(X_train, Y_train)
            train_scores.append(reg.score(X_train, Y_train))
            print("Score with Lasso with train, alpha = {} : {}".format(alpha, reg.score(X_train, Y_train)))
            print("Score with Lasso, alpha = {} : {}".format(alpha, reg.score(X_test, Y_test)))
            scores.append(reg.score(X_test, Y_test))
            count = 0
            non_zero_list = []
            non_zero_coefs_list = []
            for i, val in enumerate(reg.coef_):
                if val != 0:
                    count += 1
                    non_zero_list.append(X_train.columns.values[i])
                    non_zero_coefs_list.append(str(reg.coef_[i]))
            non_zero_number.append(count)
            non_zero_coefs.append(non_zero_coefs_list)
            non_zero_values.append(non_zero_list)

        for i, alpha in enumerate(alphas):
            decimal_split = '.'.split(str(alpha))
            ext = ''
            if len(decimal_split) > 1:
                ext = decimal_split[1]
            else:
                ext = str(alpha)
            with open('./lasso/lasso_{}.csv'.format(ext), 'wt') as file:
                csvwrite = csv.writer(file, delimiter=';')
                for j in range(non_zero_number[i]):
                    csvwrite.writerow([non_zero_values[i][j], non_zero_coefs[i][j]])

        with open('lasso.csv', 'wt') as lasso_csv:
            writer = csv.writer(lasso_csv, delimiter=';')
            writer.writerow(alphas)
            writer.writerow(train_scores)
            writer.writerow(scores)
            writer.writerow(intercepts)
            writer.writerow(non_zero_number)

        model = linear_model.Lasso()
        model.fit(X_train, Y_train)

    elif model_name == "knn":
        # hyperparamètre : nombre k de plus proches voisins
        means = []
        for w in ['uniform', 'distance']:
            for k in range(1, 10):
                neigh = KNeighborsRegressor(n_neighbors=k, weights=w)
                scores = cross_val_score(neigh, X_train, Y_train, cv=5)
                means += [[w, k, scores.mean()]]
        means.sort(key=lambda x: x[2], reverse=True)
        best_w = means[0][0]
        best_k = means[0][1]

        # apprentissage du modèle
        model = KNeighborsRegressor(n_neighbors=best_k, weights=best_w)
        model.fit(X_train, Y_train)

    elif model_name == "decision_tree":

        means = []
        for splitter in ['best', 'random']:
            for max_feature in [1, 3, 10]:
                for min_samples_s in [2, 3, 10]:
                    for min_samples_l in [1, 3, 10]:
                        tr = tree.DecisionTreeRegressor(criterion="mae", splitter=splitter, max_features=max_feature, min_samples_split=min_samples_s, min_samples_leaf=min_samples_l)
                        scores = cross_val_score(tr, X_train, Y_train, cv=5)
                        means += [[scores.mean(), splitter, max_feature, min_samples_s, min_samples_l]]
                        print([scores.mean(), splitter, max_feature, min_samples_s, min_samples_l])
        means.sort(key=lambda x: x[0], reverse=True)
        print(means)
        best_parameters=means[0]

        model = tree.DecisionTreeRegressor(criterion="mae", splitter=best_parameters[1], max_features=best_parameters[2], min_samples_split=best_parameters[3], min_samples_leaf=best_parameters[4])
        model.fit(X_train, Y_train)


    elif model_name == "grid_random_forest":

        rf = RandomForestRegressor()
        param_grid = {"max_depth": [None,15,20,30,40,50],
                      "max_features": [10, 20, 30, 50],
                      "min_samples_split": [5, 10, 20],
                      "min_samples_leaf": [1, 2, 3, 4, 5],
                      "bootstrap": [True, False],
                      "criterion": ["mae", "mse"]}

        model = GridSearchCV(rf, param_grid=param_grid)

        model.fit(X_train, Y_train)

        best_params = model.best_params_
        report(model.cv_results_, 5)
        top5 = report_export(model.cv_results_, 5)

    elif model_name == "random_random_forest":

        rf = RandomForestRegressor()
        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [None,15,20,30,40,50],
                      "max_features": sp_randint(1, 50),
                      "min_samples_split": sp_randint(5, 20),
                      "min_samples_leaf": sp_randint(1, 5),
                      "bootstrap": [True, False],
                      "criterion": ["mae", "mse"]}

        # run randomized search
        n_iter_search = 50
        model = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter_search)

        model.fit(X_train, Y_train)

        best_params = model.best_params_
        report(model.cv_results_, 5)
        top5 = report_export(model.cv_results_, 5)

    elif model_name == "random_forest_features":

        model = RandomForestRegressor(criterion="mae", max_features=40, max_depth=32, min_samples_leaf=2, min_samples_split=6, bootstrap=False)

        model.fit(X_train, Y_train)

        features=[]

        for i in range(len(model.feature_importances_)):
            features+=[[str(list(X.columns.values)[i]), model.feature_importances_[i]]]
        print(features)
        features.sort(key=lambda x: x[1], reverse=True)
        print(features)

        write_csv('feature_importance'+name, features)



    #Metrics
    mets = metrics(model, X_test, Y_test, X_train, Y_train, X_results, X_values)

    plt.plot(np.array([i for i in range(100)]), X_results['predicted_mortality'][0:100])
    plt.plot(np.array([i for i in range(100)]), X_results['true_mortality'][0:100])
    plt.savefig('../plots/predictions_fit.png')
    plt.close()
    dico = {}
    for i in range(len(X_results)):
        country = X_results.iloc[i]['area']
        year = int(X_results.iloc[i]['year'])
        true_mor = X_results.iloc[i]['true_mortality']
        if country not in dico.keys():
            dico[country] = [(year, true_mor)]
        else:
            dico[country] += [(year, true_mor)]


    # Plotting lags prediction
    # X_other = X_lag[['area', 'year', 'TOTAL_POP']]
    # X_lag = X_lag.drop(columns=['area', 'year', 'TOTAL_POP'], axis=1)
    # Y_lag = model.predict(X_lag)
    # X_lag['area'] = X_other['area']
    # X_lag['year'] = X_other['year']
    # X_lag['TOTAL_POP'] = X_other['TOTAL_POP']
    # dicolag = {}
    # for i in range(len(X_lag)):
    #     country = X_values.iloc[i]['area']
    #     year = int(X_values.iloc[i]['year'])
    #     true_mor = Y_lag[i] * X_lag.iloc[i]['TOTAL_POP']
    #     if country not in dicolag.keys():
    #         dicolag[country] = [(year, true_mor)]
    #     else:
    #         dicolag[country] += [(year, true_mor)]
    #
    # for k, v in dico.items():
    #     if len(v) > 1 :
    #         v.sort(key = lambda x : x[0])
    #     years = [year for year, _ in v]
    #     mors = [mor for _, mor in v]
    #     plt.scatter(years, mors, c='b')
    #     if k in dicolag.keys():
    #         v_lag = dicolag[k]
    #         years_lag = [year for year, _ in v_lag]
    #         mors_lag = [mor for _, mor in v_lag]
    #         plt.scatter(years_lag, mors_lag, c='r')
        # if len(years)>3:
        #     x_new = np.linspace(years[0], years[len(years)-1], 300)
        #     # print(years)
        #     # print(mors)
        #     # print(x_new)
        #     mors_smooth = spline(years, mors, x_new)
        #     plt.plot(x_new, mors_smooth)
        # plt.savefig('../plots/' + k + '.png')
        # plt.close()



    #Exporting results
    if model_name == 'ridge_regression' or model_name == 'lasso_regression':

        mets['alpha'] = best_alpha
        results = pd.DataFrame(data=mets,
                               index=[0])
        results.to_csv(model_name + '_' + name + '_results.csv', index=False)

    elif model_name == 'knn':

        mets['k'] = best_k
        results = pd.DataFrame(data=mets,
                               index=[0])
        results.to_csv(model_name + '_' + name + '_results.csv', index=False)

    elif model_name == 'random_random_forest':

        results = pd.DataFrame(columns= ['R2_train', 'R2_test', 'MSE', 'RMSE', 'MAE', 'MPE', 'MAPE', 'MD'])
        parameters = pd.DataFrame(columns= ['max_depth','max_features', 'min_samples_split', 'min_samples_leaf','bootstrap','criterion'])

        for i in range(1,6):
            print("rank : ()".format(i))
            params = top5[i]
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, Y_train)
            scores = metrics(rf,X_test, Y_test, X_train, Y_train, X_results, X_values)
            results_temp = pd.DataFrame(data=scores,
                                   index=[0])
            params_temp = pd.DataFrame(data = params, index=[0])
            results = results.append(results_temp)
            parameters = parameters.append(params_temp)

        if developing_countries:
            model_name += '_PVD'

        results.to_csv(model_name + '_' + name + '_results.csv', index=False)
        parameters.to_csv(model_name + '_' + name + '_params.csv', index= False)


    else:
        results = pd.DataFrame(data = mets, index = [0])
        results.to_csv(model_name +'_' + name + '_results.csv', index = False)

def write_csv(name, rows):
    with open(name + '.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        for row in rows:
            writer.writerow(row)

# 5 meilleurs résulats cross validation
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def report_export(results, n_top = 5):
    res = {}
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            rank = i
            parameters = results['params'][candidate]
            res[rank] = parameters
    return res


def remove_outliers(df):
    indexes = []
    d = {col_name: df[col_name] for col_name in df.columns.values}
    df = pd.DataFrame(data=d)
    df = df.reset_index(drop=True)
    outliers = [('Brazil', 1977), ('Brazil', 1978), ('Colombia', 1981), ('Haiti', 1981), ('Haiti', 1983), ('Honduras', 1982), ('Honduras', 1983), ('Jamaica', 1968), ('Jamaica', 1969), ('Jamaica', 1970), ('Jamaica', 1971), ('Jamaica', 1975), ('Pakistan', 1993), ('Pakistan', 1994), ('Portugal', 2004), ('Portugal', 2005), ('Puerto Rico', 1979), ('Bolivia', 2002), ('Azerbaijan', 2003), ('Grenada', 1974), ('Grenada', 1975), ('Grenada', 1976), ('Grenada', 1977), ('Guadeloupe', 1971), ('Guadeloupe', 1972), ('Guadeloupe', 1973), ('Guadeloupe', 1976), ('Guadeloupe', 1977), ('Guadeloupe', 1978), ('Guadeloupe', 1979), ('Guadeloupe', 1980), ('San Marino', 2011), ('San Marino', 2012), ('San Marino', 2013), ('San Marino', 2014), ('San Marino', 2015)]
    for i in range(df.shape[0]):
        for outlier in outliers:
            if df.iloc[i]['area'] == outlier[0] and df.iloc[i]['year'] == outlier[1]:
                #print("Found {} {}".format(outlier[0], outlier[1]))
                indexes += [i]
    df.drop(df.index[indexes], inplace = True, axis=0)
    return df


def metrics(model, X_test, Y_test, X_train, Y_train, X_results, X_values):
    # Computes main metrics for considered model
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
    max_one_true = np.maximum(np.ones(len(X_results['true_mortality'])), X_results['true_mortality'])
    division = np.divide(absdiff, max_one_true)
    div = list(zip(list(X_results['area']), list(X_results['year']), list(division), list(absdiff), list(max_one_true)))
    div.sort(key=lambda x: x[2])
    div.sort(key=lambda x: (x[0], x[1]))
    div = [('Country', 'year', 'APE', 'AE', 'GT')] + div
    write_csv('../plots/evolution_per_country', div)
    mape_test = np.mean(division)
    dic['MAPE'] = mape_test
    print("Mean Absolute Percentage of Error : %s" % mape_test)
    # Mean Deviation
    mean = np.mean(X_results['true_mortality'])
    md = np.mean(abs(X_results['true_mortality'] - mean))
    dic['MD'] = md
    print("Mean deviation : %s" % md)

    return dic

def lag_X_Y(df):
    X_lag = df.loc[df['relative_mortality'] == 0]
    X = df.loc[df['relative_mortality'] != 0]
    X = X.drop(columns=['area', 'year', 'relative_mortality', 'TOTAL_POP', 'Unnamed: 0'], axis=1)
    X_lag = X_lag.drop(columns=['relative_mortality', 'Unnamed: 0'], axis=1)
    Y = df.loc[df['relative_mortality'] != 0].relative_mortality
    return X_lag, X, Y

predict_mortality("ALL_MV50_VT_Merged", "random_random_forest", "C16", developing_countries= True,  test_size= 0.33)
