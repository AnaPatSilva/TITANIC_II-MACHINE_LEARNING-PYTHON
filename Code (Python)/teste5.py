# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:12:15 2023

@author: anapa
"""

##% Com pipeline e categorical (if binary) e numeric transformers
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

#% 4.2 Definir pré-processamento para os dados numéricos: Simple Imputer (substituir NANs)
# Como não tenho NANs vou ignorar este passo
numerical_transformer = SimpleImputer(strategy='constant') #strategy='constant': preenche os valores ausentes com um valor constante especificado por outro parâmetro chamado fill_value.


#% 4.3 Definir pré-processamento para os dados categóricos: Simple Imputer e onehot encoder - Pipeline
# Como não tenho NANs vou alterar o código
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), #preenche os valores ausentes nas colunas categóricas com o valor mais frequente (moda) presente em cada coluna
    ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore')) #transforma as variáveis categóricas em uma representação binária (0 ou 1) para cada categoria possível
])


#% 4.4 Criar pré-processador para a transformação das colunas tendo em conta as definições anteriores
# Como não tenho alterações a fazer nas features numéricas, vou alterar o código
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num),
        ('cat', categorical_transformer, cat)
    ])


#% 4.5 Definir o(s) modelo(s) de ML (Decision Tree, Random Forest ou XGBoost)
modelo_RF = RandomForestClassifier(n_estimators=100, random_state=6)


#% 4.6 Criar a pipeline com o pré-processador e o modelo
RF = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', modelo_RF)
                             ])

#% 4.7 Fit do modelo
RF.fit(x_train, y_train)


#% Ver os dados transformados (pelo preprocessor)
x_train_transformed = preprocessor.transform(x_train)
trans_train = pd.DataFrame(x_train_transformed) #colocar as colunas na ordem em que foram processadas (primeiro as numéricas e depois as categóricas)
trans_train.head()


#% 4.8 Previsões do modelo
previsões = RF.predict(x_test)

# Evaluate the model
score = mean_absolute_error(y_test, previsões)
print('MAE:', score)

#0.1564245810055866
#igual ao teste 3 e 4