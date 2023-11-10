# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 12:45:35 2023

@author: anapa
"""

##% Sem pipeline e categorical onehot
## Random Forest Classifier

import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error


#% 1.2 Importação de datasets
for dirname, _, filenames in os.walk('G:\O meu disco\Formação\Kaggle\Kaggle Titanic\Data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('G:\O meu disco\Formação\Kaggle\Kaggle Titanic\Data\\train.csv')

y = train['Survived']
features = ['Sex', 'Pclass', 'Fare']
x = train[features]


#% 3.2 Dividir o dataset em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, test_size=0.2,
                                                    random_state=6)


#%% 4. Prepare for ML

#% 4.1 Definir as colunas categóricas e numéricas
cat = ['Sex']
#OU cat = [cname for cname in x_train.columns if x_train[cname].nunique() < 10 and x_train[cname].dtype == "object"]
num = ['Pclass', 'Fare']
#OU num = [cname for cname in x_train.columns if x_train[cname].dtype in ['int64', 'float64']]


#% 4.2 OneHot Encoder das variáveis categóricas
OH_encoder = OneHotEncoder(drop='if_binary', #drop='if_binary' não cria 2 colunas distintas se a categoria for binária (ex: male, female)
                           handle_unknown='ignore', #handle_unknow ='ignore' ignora categorias que não têm valores
                           sparse_output=False)

# PARA OS DADOS DE TREINO
# Transforma os dados e cria um DataFrame com as colunas codificadas
cat_encoded_train = pd.DataFrame(OH_encoder.fit_transform(x_train[cat]),
                          columns=OH_encoder.get_feature_names_out(input_features=['Sex'])) #transforma as colunas e mantém os seus nomes, caso contrário passariam a números, sendo impossível de distinguir

print(cat_encoded_train.head())

# O Onehot remove o index, por isso temos de o voltar a colocar
cat_encoded_train.index = x_train.index

# Remover as colunas categóricas, para depois as substituir pelas colunas onehot
x_train = x_train.drop(cat, axis=1)
x_train = pd.concat([x_train, cat_encoded_train], axis=1)

# Garantir que todas as colunas são do tipo string
x_train.columns = x_train.columns.astype(str)

print(x_train.head())


# PARA OS DADOS DE TESTE
# Transforma os dados e cria um DataFrame com as colunas codificadas
cat_encoded_test = pd.DataFrame(OH_encoder.fit_transform(x_test[cat]),
                          columns=OH_encoder.get_feature_names_out(input_features=['Sex'])) #transforma as colunas e mantém os seus nomes, caso contrário passariam a números, sendo impossível de distinguir

print(cat_encoded_test.head())

# O Onehot remove o index, por isso temos de o voltar a colocar
cat_encoded_test.index = x_test.index

# Remover as colunas categóricas, para depois as substituir pelas colunas onehot
x_test = x_test.drop(cat, axis=1)
x_test = pd.concat([x_test, cat_encoded_test], axis=1)

# Garantir que todas as colunas são do tipo string
x_test.columns = x_test.columns.astype(str)

print(x_test.head())


#% 4.3 Definir o(s) modelo(s) de ML (Decision Tree, Random Forest ou XGBoost)
RF = RandomForestClassifier(n_estimators=100, random_state=6)

#% 4.4 Fit do modelo
RF.fit(x_train, y_train)


#% 4.8 Previsões do modelo
previsões = RF.predict(x_test)

# Evaluate the model
score = mean_absolute_error(y_test, previsões)
print('MAE:', score)

#0.1564245810055866