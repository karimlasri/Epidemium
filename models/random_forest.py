from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from categorical_encoder import categorical_encoding
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv("C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/ALL_PCA2_50.csv")

df = df.loc[df['type'] == "C16"]

df = categorical_encoding(df)

Y = df['relative_mortality']
Y = (Y - Y.min())/(Y.max()-Y.min())

X = df.drop(columns = ['relative_mortality','year'])

results = []

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 20, 30, 50]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 5, 10, 15, 20, 30, 50, 70]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X, Y)

rf_random.best_params_