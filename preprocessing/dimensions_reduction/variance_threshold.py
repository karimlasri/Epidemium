from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler


def variance_threshold(df, thresh):
    #Enlever les colonnes de catégories et les remettre à la fin
    #df = pd.read_csv("Worldbank_Replaced_Countries.csv")
    #print(list(df.columns.values))
    #print(len(list(df.columns.values)))
    # Columns that we want to keep afterwards


    # Replacing values by mean
    df.fillna(df.mean(), inplace=True)


    # Standard scaling before running variance threshold
    min_max_scaler = MinMaxScaler()
    imputed_array = min_max_scaler.fit_transform(df)

    columns = df.columns

    selector = VarianceThreshold(threshold=thresh)
    selector.fit_transform(imputed_array)

    labels = [columns[x] for x in selector.get_support(indices=True)]

    imputed_df = pd.DataFrame(selector.fit_transform(imputed_array), columns=labels) #, columns = df.columns)
    final_df = imputed_df.reset_index(drop=True)

    return(final_df)