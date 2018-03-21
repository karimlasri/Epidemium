import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Imputer

# df = pd.read_csv("C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/WORLDBANK50.csv")
# df = df.drop(columns=['Unnamed: 0', 'area_code', 'area'])
# df = df.fillna(0)
# print(list(df))


def transform_pca(df):

    """ Function to compute the new dataset from PCA
        It assumes that all parameters are numerical and that there are no missing values
    """

    #Replacing missing values
    values = df.values
    imputer = Imputer()
    values = imputer.fit_transform(values)

    #Rescaling data
    values_ss = StandardScaler().fit(values)
    values_std = values_ss.transform(values)

    VARIANCE_TO_KEEP = 0.9

    #Computing PCA to determine best n_components to fit VARIANCE_TO_KEEP
    sklearn_pca = PCA()
    Y_sklearn = sklearn_pca.fit_transform(values_std)

    #plotting the explained variance
    plt.bar(range(sklearn_pca.n_components_), sklearn_pca.explained_variance_ratio_, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(sklearn_pca.n_components_),
             [sklearn_pca.explained_variance_ratio_[:y].sum() for y in range(1, sklearn_pca.n_components_ + 1)],
             alpha=0.5, where='mid', label='cumulative explained variance')
    plt.axhline(y=0.95, linewidth=2, color='r')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.xlim([0, sklearn_pca.n_components_])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


    #recommendation for how many vectors to keep
    keep_recommend = [sklearn_pca.explained_variance_ratio_[:y].sum() > VARIANCE_TO_KEEP for y in
                      range(1, sklearn_pca.n_components_ + 1)].count(False)
    print("Number of components to keep to retain %s%% of the variance:" % (100 * VARIANCE_TO_KEEP),
          keep_recommend, "out of the original", sklearn_pca.n_components_)


    #Computing final PCA
    sklearn_pca_final = PCA(n_components = keep_recommend)
    Y_sklearn_final = sklearn_pca_final.fit_transform(values_std)

    #
    print(type(Y_sklearn_final))

    #Creating new dataframe
    df_PCA = pd.DataFrame(Y_sklearn_final,
                          columns=[("PCA_component_" + str(comp)) for comp in range(sklearn_pca_final.n_components)])

    return df_PCA

# transform_pca(df)
