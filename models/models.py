import pandas as pd, numpy as np
from IPython.display import display
from IPython.display import Image
import math
import sklearn
import sklearn.cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import random

import matplotlib.pyplot as plt
import csv

from sklearn.linear_model import LinearRegression
from sklearn import linear_model

from sklearn.neighbors import KNeighborsRegressor

from sklearn import tree

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint



def predict_mortality(df, model, cancer_type, test_size, developing_countries=False):

    # if developing_countries:
    #     #sélection des pays en voie de développement
    #     countries_df = pd.read_csv('developping_countries.csv')
    #     countries=countries_df['area']
    #     df=df[df.area.isin(countries)]

    df = df.dropna(subset = ['TOTAL_POP'], axis = 0)

    #sélection du type de cancer
    df=df[df.type == cancer_type]
    df=df.drop('type', axis=1)

    #variable à prédire : mortalité relative
    X=df.drop(columns = ['area', 'year', 'relative_mortality', 'TOTAL_POP', 'Unnamed: 0'], axis=1)
    Y=df.relative_mortality

    #standardisation des variables d'entrée pour les modèles linéaires
    if model in ["linear_regression", "ridge_regression", "lasso_regression"]:
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

    X_test=X_test.drop('TOTAL_POP', axis=1)
    X_train=X_train.drop('TOTAL_POP', axis=1)

    X_test=X_test.drop('area', axis=1)
    X_train=X_train.drop('area', axis=1)

    X_test=X_test.drop('year', axis=1)
    X_train=X_train.drop('year', axis=1)


    if model=="linear_regression":
        #pas d'hyperparamètre à déterminer
        #apprentissage du modèle
        model = linear_model.LinearRegression()
        model.fit(X_train, Y_train)
        print("Coefs : {}".format(list(zip(X_train.columns.values, model.coef_))))

    elif model == "ridge_regression":

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

        model = linear_model.Ridge(alpha = best_alpha)
        model.fit(X_train, Y_train)


    elif model == "lasso_regression":

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

    elif model == "knn":
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

    elif model == "decision_tree":

        means = []
        for criterion in ['mse', 'friedman_mse', 'mae']:
            for splitter in ['best', 'random']:
                tr = tree.DecisionTreeRegressor(criterion=criterion, splitter=splitter)
                scores = cross_val_score(tr, X_train, Y_train, cv=5)
                means += [[criterion, splitter, scores.mean()]]
        means.sort(key=lambda x: x[2], reverse=True)
        best_criterion = means[0][0]
        best_splitter = means[0][1]

        print("best criterion %s" % best_criterion)
        print("best splitter %s " % best_splitter)

        model = tree.DecisionTreeRegressor(criterion=best_criterion, splitter=best_splitter)
        model.fit(X_train, Y_train)

    elif model == "random_forest":

        means = []
        for criterion in ['mse', 'mae']:
            for n_estimators in range(10, 15):
                rf = RandomForestRegressor(criterion=criterion, n_estimators=n_estimators)
                scores = cross_val_score(rf, X_train, Y_train, cv=5)
                means += [[criterion, n_estimators, scores.mean()]]
        means.sort(key=lambda x: x[2], reverse=True)
        best_criterion = means[0][0]
        best_n_estimators = means[0][1]
        print("best_criterion %s" % best_criterion)
        print("best n_estimators %s " % best_n_estimators)

        model = RandomForestRegressor(criterion=best_criterion, n_estimators=best_n_estimators)
        model.fit(X_train, Y_train)

    elif model == "random_forest_2":

        rf = RandomForestRegressor()
        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [3, None],
                      "max_features": sp_randint(1, 11),
                      "min_samples_split": sp_randint(2, 11),
                      "min_samples_leaf": sp_randint(1, 11),
                      "bootstrap": [True, False],
                      "criterion": ["mae", "mse"]}

        # run randomized search
        n_iter_search = 20
        model = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter_search)

        model.fit(X_train, Y_train)

    elif model == "random_forest_3":


        n_iter_search = 10
        means = []
        for i in range(n_iter_search):
            max_d = None
            max_f = random.randint(16, 25)
            min_samples_s = random.randint(2, 20)
            min_samples_l = random.randint(1, 4)
            bootstr = False

        # run randomized search

            model = RandomForestRegressor(criterion='mae', max_depth=max_d, min_samples_split=min_samples_s, min_samples_leaf=min_samples_l, max_features=max_f, bootstrap=bootstr)
            scores = cross_val_score(model, X_train, Y_train, cv=5)
            print(max_d, max_f, min_samples_s, min_samples_l, bootstr, scores.mean())
            means += [[max_d, max_f, min_samples_s, min_samples_l, bootstr, scores.mean()]]

        means.sort(key=lambda x : x[5], reverse = True)
        print(means)
        best_params = means[0][:5]
        best_means = means[0][5]
        print(means[0])

        model = RandomForestRegressor(criterion='mae', max_depth=best_params[0], min_samples_split=best_params[2],
                                      min_samples_leaf=best_params[3], max_features=best_params[1], bootstrap=best_params[4])
        model.fit(X_train, Y_train)

    elif model == "random_forest_4":

        means = []
        rows = []
        header = ['']+[i for i in range(16, 26, 2)]
        rows += [header]
        for max_f in range(16, 26, 2):
            row = [max_f]
            for min_samples_s in range(2, 21, 2):
                    model = RandomForestRegressor(criterion='mae', max_depth=None, min_samples_split=min_samples_s,
                                                  min_samples_leaf=1, max_features=max_f, bootstrap=False)
                    scores = cross_val_score(model, X_train, Y_train, cv=5)
                    print(max_f, min_samples_s, scores.mean())
                    means += [[max_f, min_samples_s, scores.mean()]]
                    row+= [scores.mean()]
            rows += [row]
        means.sort(key=lambda x: x[2], reverse=True)
        print(means)
        best_params = means[0][:2]
        print(means[0])

        model = RandomForestRegressor(criterion='mae', max_depth=None, min_samples_split=best_params[1],
                                      min_samples_leaf=1, max_features=best_params[0],
                                      bootstrap=False)
        model.fit(X_train, Y_train)

        write_csv('random_forest_grid', rows)
        # for i in range(len(tr.feature_importances_)):
        #     print("{}".format(str(list(X.columns.values)[i])))
        #     print("{}".format(str(tr.feature_importances_[i])))

    # coefficient de détermination
    print("Coefficient of Determination %s" % model.score(X_test, Y_test))
    # prédiction de la mortalité en volume à partir de la mortalité relative prédite par le modèle
    Y_predicted = model.predict(X_test)
    X_results['predicted_relative_mortality'] = Y_predicted
    X_results['relative_mortality'] = Y_test
    X_results['predicted_mortality'] = X_results['predicted_relative_mortality'] * X_results['TOTAL_POP']
    X_results['predicted_mortality'] = X_results['predicted_mortality'].round()
    X_results['true_mortality'] = X_results['relative_mortality'] * X_results['TOTAL_POP']
    # Mean Square Error
    mse_test = np.mean((X_results['true_mortality'] - X_results['predicted_mortality']) ** 2)
    print("Mean Square Error : %s" % mse_test)
    # Root Mean Square Error
    rmse_test = math.sqrt(mse_test)
    print("Root Mean Square Error : %s" % rmse_test)
    # Mean Average Error
    mae_test = np.mean(abs(X_results['true_mortality'] - X_results['predicted_mortality']))
    print("Mean Average Error : %s" % mae_test)
    # Relative Mean Average Error
    mean_mortality_test = np.mean(X_results['true_mortality'])
    rel_mae = mae_test / mean_mortality_test
    print("Mean Percentage of Error : %s" % rel_mae)


def write_csv(name, rows):
    with open(name + '.csv', 'wt') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        for row in rows:
            writer.writerow(row)


dataframe = pd.read_csv('../datasets/final_datasets/ALL_MV30_VT_Merged.csv')

predict_mortality(dataframe, 'ridge_regression', 'C16', 0.33)