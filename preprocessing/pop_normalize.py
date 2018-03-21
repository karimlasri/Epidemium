import pandas as pd

df = pd.read_csv('C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/WB_mort.csv')

mort = df['sum']
pop = df['SP.POP.TOTL']

df['relative_mortality'] = mort/pop

df.to_csv('C:/Users/titou/Desktop/Centrale/Option OSY/Projet/Datasets/WB_mort_norm.csv')