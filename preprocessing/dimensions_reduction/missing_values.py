import pandas as pd
import numpy as np

#Répartition des colonnes en fonction de leur % de valeurs manquantes pour un fichier csv donné
# NOT USED
def columns_repartition(file_name):
    data_df = pd.read_csv(file_name)
    col_names = list(data_df.columns.values)
    print(col_names)
    number_valid_columns = np.zeros(20)
    number_valid_columns_int=[int(i) for i in number_valid_columns]
    percentages_list=[]
    for l in range(20):
        percentages_list+=[str(l*5)+'%']
    for i in range(20):
        percentage=((1/20)*i)
        for col_name in col_names:
            col = data_df[col_name]
            nb_nan = col.isnull().sum()
            if (nb_nan / len(col)) <= percentage :
                number_valid_columns_int[i]+=1
    print(number_valid_columns_int)
    print(percentages_list)
    with open(str(file_name)[:len(file_name)-4] + '_col_repart.csv', 'w') as file:
        for i in range(20):
            file.write(percentages_list[i] + ';' + str(number_valid_columns_int[i]) + '\n')


#Retourne le dataset wb en gardant les colonnes avec un pourcentage de valeurs manquantes inférieur au pourcentage demandé
def clean_columns_df(data_df, percentage):
    #data_df = pd.read_csv('./Worldbank_Replaced_Countries.csv')
    col_names = list(data_df.columns.values)
    feature_list=[]

    for col_name in col_names:
            col = data_df[col_name]
            nb_nan = col.isnull().sum()
            if (nb_nan / len(col)) <= percentage :
                feature_list+=[col_name]
    df1 = data_df[feature_list]
    #df1.to_csv('./Worldbank_Replaced_Countries_Var30.csv')
    return(df1)
