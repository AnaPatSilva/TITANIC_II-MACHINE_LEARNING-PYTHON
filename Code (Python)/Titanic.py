# -*- coding: utf-8 -*-
"""
Nova submissão para a competição Titanic do Kaggle
"""

#%% 1. Import Data & Libraries

#% 1.1 Instalação de livrarias
import os
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix

#% 1.2 Importação de datasets
for dirname, _, filenames in os.walk('G:\O meu disco\Formação\Kaggle\Kaggle Titanic\Data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

gender = pd.read_csv('G:\O meu disco\Formação\Kaggle\Kaggle Titanic\Data\gender_submission.csv')
test = pd.read_csv('G:\O meu disco\Formação\Kaggle\Kaggle Titanic\Data\\test.csv')
train = pd.read_csv('G:\O meu disco\Formação\Kaggle\Kaggle Titanic\Data\\train.csv')

print(gender.head())
print(test.head())
print(train.head())

print(train.shape)
train.describe()


#%% 2. Exploratory Data Analysis

#% 2.1 Data Profiling do dataset de treino (comentado para não gerar relatório sempre que corro o código)
#profile_train = ProfileReport(train, title="Profile Train")
#profile_train.to_file("profile_train.html")

#% 2.2 Fazer uma Matriz de Correlação
train.dtypes #verificar o tipo de dados de cada coluna
columns_drop = ['Name', 'Ticket', 'Cabin'] #vou eliminar estas colunas que no data profilling constam como "text"
train = train.drop(columns=columns_drop)

# Para conseguir fazer a matriz de correlação tenho de converter as variáveis categóricas em numéricas
# Vou fazer esta conversão agora apenas para a matriz, através de um onehot coding
object = (train.dtypes == 'object')
object = list(object[object].index)
print("Categorical variables:")
print(object)

# Aplicar o onehot a cada coluna categórica
OH_encoder = OneHotEncoder(drop='if_binary', #drop='if_binary' não cria 2 colunas distintas se a categoria for binária (ex: male, female)
                           handle_unknown='ignore', #handle_unknow ='ignore' ignora categorias que não têm valores
                           sparse_output=False) #sparse_output = False para garantir que os dados saem em Numpy
OH_columns = pd.DataFrame(OH_encoder.fit_transform(train[object]),
                          columns=OH_encoder.get_feature_names_out(input_features=object)) #transforma as colunas e mantém os seus nomes, caso contrário passariam a números, sendo impossível de distinguir

# O Onehot remove o index, por isso temos de o voltar a colocar
OH_columns.index = train.index

# Remover as colunas categóricas, para depois as substituir pelas colunas onehot
train_onehot = train.drop(object, axis=1)
train_onehot = pd.concat([train_onehot, OH_columns], axis=1)

# Garantir que todas as colunas são do tipo string
train_onehot.columns = train_onehot.columns.astype(str)

# Fazer a Matriz de Correlação
correlation_matrix = train_onehot.corr()
print(correlation_matrix)

# Criar um mapa de calor da matriz de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação')
plt.show()

# Obter as correlações do target 'Survived' com todas as outras features
correlations_target = correlation_matrix['Survived']

# Remover a correlação com a própria variável 'Survived' (que será 1)
correlations_target = correlations_target.drop('Survived')

# Obter as 3 features mais correlacionadas com 'Survived' em ordem decrescente
top_features = correlations_target.abs().sort_values(ascending=False).head(3)
print("As features mais correlacionadas com 'Survived':")
for feature_name in top_features.index:
    correlation_value = top_features[feature_name]
    print(f"{feature_name} {correlation_value:.2f}")

# Há uma correlação grande entre Survived (target) e Sex, Pclass e Fare (features)

#% 2.3 Verificar se há NANs nas variáveis a utilizar (Survived, Sex, Pclass e Fare)
NAN_Survived = train['Survived'].isnull()
print('Número de valores nulos em "Survived"', {NAN_Survived.sum()})
NAN_Sex = train['Sex'].isnull()
print('Número de valores nulos em "Sex"', {NAN_Sex.sum()})
NAN_Pclass = train['Pclass'].isnull()
print('Número de valores nulos em "Pclass"', {NAN_Pclass.sum()})
NAN_Fare = train['Fare'].isnull()
print('Número de valores nulos em "Fare"', {NAN_Fare.sum()})

# Não há missing values


#%% 3. Train/Test Split

#% 3.1 Definir o target (Survived) e as features (Sex, Pclass, Fare)
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


#% Diversas tentativas de aplicação do modelo, avaliando no final através do MAE

#%% 1) Com pipeline e categorical e numeric transformers
## Random Forest Regression e Random Forest Classifier

#% 4.2 Definir pré-processamento para os dados numéricos: Simple Imputer (substituir NANs)
#numerical_transformer = SimpleImputer(strategy='constant') #strategy='constant': preenche os valores ausentes com um valor constante especificado por outro parâmetro chamado fill_value.


#% 4.3 Definir pré-processamento para os dados categóricos: Simple Imputer e onehot encoder - Pipeline
#categorical_transformer = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='most_frequent')), #preenche os valores ausentes nas colunas categóricas com o valor mais frequente (moda) presente em cada coluna
#    ('onehot', OneHotEncoder(handle_unknown='ignore')) #transforma as variáveis categóricas em uma representação binária (0 ou 1) para cada categoria possível
#])


#% 4.4 Criar pré-processador para a transformação das colunas tendo em conta as definições anteriores
#preprocessor = ColumnTransformer(
#    transformers=[
#        ('num', numerical_transformer, num),
#        ('cat', categorical_transformer, cat)
#    ])


#% 4.5 Definir o(s) modelo(s) de ML (Decision Tree, Random Forest ou XGBoost)
#modelo_RFR = RandomForestRegressor(n_estimators=100, random_state=6)
#modelo_RFC = RandomForestClassifier(n_estimators=100, random_state=6)


#% 4.6 Criar a pipeline com o pré-processador e os modelos
#RFR = Pipeline(steps=[('preprocessor', preprocessor),
#                              ('model', modelo_RFR)
#                             ])
#RFC = Pipeline(steps=[('preprocessor', preprocessor),
#                              ('model', modelo_RFC)
#                             ])


#% 4.7 Fit do modelo
#RFR.fit(x_train, y_train)
#RFC.fit(x_train, y_train)


#% 4.8 Ver os dados transformados (pelo preprocessor)
# x_train_transformed = preprocessor.transform(x_train)
# trans_train = pd.DataFrame(x_train_transformed) #colocar as colunas na ordem em que foram processadas (primeiro as numéricas e depois as categóricas)
# trans_train.head()


#% 4.9 Previsões do modelo
# previsões_RFR = RFR.predict(x_test)
# previsões_RFC = RFC.predict(x_test)

#% 5.0 Avaliar o modelo
# score_RFR = mean_absolute_error(y_test, previsões_RFR)
# print('MAE:', score_RFR)
# #0.20492569657072424
# score_RFC = mean_absolute_error(y_test, previsões_RFC)
# print('MAE:', score_RFC)
#0.1564245810055866


#%% 2) Com pipeline e categorical (if binary) e numeric transformers
## Random Forest Regression e Random Forest Classifier

#% 4.2 Definir pré-processamento para os dados numéricos: Simple Imputer (substituir NANs)
# numerical_transformer = SimpleImputer(strategy='constant') #strategy='constant': preenche os valores ausentes com um valor constante especificado por outro parâmetro chamado fill_value.


#% 4.3 Definir pré-processamento para os dados categóricos: Simple Imputer e onehot encoder - Pipeline
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')), #preenche os valores ausentes nas colunas categóricas com o valor mais frequente (moda) presente em cada coluna
#     ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore')) #transforma as variáveis categóricas em uma representação binária (0 ou 1) para cada categoria possível
# ])


#% 4.4 Criar pré-processador para a transformação das colunas tendo em conta as definições anteriores
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, num),
#         ('cat', categorical_transformer, cat)
#     ])


#% 4.5 Definir o(s) modelo(s) de ML (Decision Tree, Random Forest ou XGBoost)
#modelo_RFR = RandomForestRegressor(n_estimators=100, random_state=6)
# modelo_RFC = RandomForestClassifier(n_estimators=100, random_state=6)

#% 4.6 Criar a pipeline com o pré-processador e o modelo
#RFR = Pipeline(steps=[('preprocessor', preprocessor),
#                              ('model', modelo_RFR)
#                             ])
# RFC = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', modelo_RFC)
#                              ])


#% 4.7 Fit do modelo
#RFR.fit(x_train, y_train)
# RFC.fit(x_train, y_train)


#% 4.8 Ver os dados transformados (pelo preprocessor)
# x_train_transformed = preprocessor.transform(x_train)
# trans_train = pd.DataFrame(x_train_transformed) #colocar as colunas na ordem em que foram processadas (primeiro as numéricas e depois as categóricas)
# trans_train.head()


#% 4.9 Previsões do modelo
#previsões_RFR = RFR.predict(x_test)
# previsões_RFC = RFC.predict(x_test)

#% 5.0 Avaliar o modelo
#score_RFR = mean_absolute_error(y_test, previsões_RFR)
#print('MAE:', score_RFR)
#0.20508371418388477
# score_RFC = mean_absolute_error(y_test, previsões_RFC)
# print('MAE:', score_RFC)
#0.1564245810055866


#%% 3) Com pipeline e categorical transformers
## Random Forest Regression e Random Forest Classifier

#% 4.2 Definir pré-processamento para os dados categóricos: Simple Imputer e onehot encoder - Pipeline
# Como não tenho NANs vou alterar o código
# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore')) #transforma as variáveis categóricas em uma representação binária (0 ou 1) para cada categoria possível
# ])

#% 4.3 Criar pré-processador para a transformação das colunas tendo em conta as definições anteriores
# Como não tenho alterações a fazer nas features numéricas, vou alterar o código
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', categorical_transformer, cat)
#     ])


#% 4.4 Definir o(s) modelo(s) de ML (Decision Tree, Random Forest ou XGBoost)
# modelo_RFR = RandomForestRegressor(n_estimators=100, random_state=6)
# modelo_RFC = RandomForestClassifier(n_estimators=100, random_state=6)


#% 4.5 Criar a pipeline com o pré-processador e o modelo
# RFR = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', modelo_RFR)
#                               ])
# RFC = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', modelo_RFC)
#                               ])

#% 4.6 Fit do modelo
# RFR.fit(x_train, y_train)
# RFC.fit(x_train, y_train)


#% 4.7 Ver os dados transformados (pelo preprocessor)
# x_train_transformed = preprocessor.transform(x_train)
# trans_train = pd.DataFrame(x_train_transformed) #colocar as colunas na ordem em que foram processadas (primeiro as numéricas e depois as categóricas)
# trans_train.head()


#% 4.8 Previsões do modelo
# previsões_RFR = RFR.predict(x_test)
# previsões_RFC = RFC.predict(x_test)

# 4.9 Avaliar o modelo
# score_RFR = mean_absolute_error(y_test, previsões_RFR)
# print('MAE:', score_RFR)
#0.3178004807252373
# score_RFC = mean_absolute_error(y_test, previsões_RFC)
# print('MAE:', score_RFC)
#0.1787709497206704


#%% 4) Sem pipeline e categorical onehot
## Random Forest Classifier

#% 4.2 OneHot Encoder das variáveis categóricas
# OH_encoder = OneHotEncoder(drop='if_binary', #drop='if_binary' não cria 2 colunas distintas se a categoria for binária (ex: male, female)
#                            handle_unknown='ignore', #handle_unknow ='ignore' ignora categorias que não têm valores
#                            sparse_output=False)

# PARA OS DADOS DE TREINO
# Transforma os dados e cria um DataFrame com as colunas codificadas
# cat_encoded_train = pd.DataFrame(OH_encoder.fit_transform(x_train[cat]),
#                           columns=OH_encoder.get_feature_names_out(input_features=['Sex'])) #transforma as colunas e mantém os seus nomes, caso contrário passariam a números, sendo impossível de distinguir

# print(cat_encoded_train.head())

# O Onehot remove o index, por isso temos de o voltar a colocar
# cat_encoded_train.index = x_train.index

# Remover as colunas categóricas, para depois as substituir pelas colunas onehot
# x_train = x_train.drop(cat, axis=1)
# x_train = pd.concat([x_train, cat_encoded_train], axis=1)

# Garantir que todas as colunas são do tipo string
# x_train.columns = x_train.columns.astype(str)

# print(x_train.head())


# PARA OS DADOS DE TESTE
# Transforma os dados e cria um DataFrame com as colunas codificadas
# cat_encoded_test = pd.DataFrame(OH_encoder.fit_transform(x_test[cat]),
#                           columns=OH_encoder.get_feature_names_out(input_features=['Sex'])) #transforma as colunas e mantém os seus nomes, caso contrário passariam a números, sendo impossível de distinguir

# print(cat_encoded_test.head())

# O Onehot remove o index, por isso temos de o voltar a colocar
# cat_encoded_test.index = x_test.index

# Remover as colunas categóricas, para depois as substituir pelas colunas onehot
# x_test = x_test.drop(cat, axis=1)
# x_test = pd.concat([x_test, cat_encoded_test], axis=1)

# Garantir que todas as colunas são do tipo string
# x_test.columns = x_test.columns.astype(str)

# print(x_test.head())


#% 4.3 Definir o(s) modelo(s) de ML (Decision Tree, Random Forest ou XGBoost)
# RF = RandomForestClassifier(n_estimators=100, random_state=6)


#% 4.4 Fit do modelo
# RF.fit(x_train, y_train)


#% 4.5 Previsões do modelo
# previsões = RF.predict(x_test)


# 4.6 Avaliar o modelo
# score = mean_absolute_error(y_test, previsões)
# print('MAE:', score)
#0.1564245810055866


#%% 5) Com pipeline e categorical (if binary) e numeric transformers
## XGBoost

#% 4.2 Definir pré-processamento para os dados numéricos: Simple Imputer (substituir NANs)
numerical_transformer = SimpleImputer(strategy='constant') #strategy='constant': preenche os valores ausentes com um valor constante especificado por outro parâmetro chamado fill_value.


#% 4.3 Definir pré-processamento para os dados categóricos: Simple Imputer e onehot encoder - Pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), #preenche os valores ausentes nas colunas categóricas com o valor mais frequente (moda) presente em cada coluna
    ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore')) #transforma as variáveis categóricas em uma representação binária (0 ou 1) para cada categoria possível
])


#% 4.4 Criar pré-processador para a transformação das colunas tendo em conta as definições anteriores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num),
        ('cat', categorical_transformer, cat)
    ])


#% 4.5 Definir o(s) modelo(s) de ML (Decision Tree, Random Forest ou XGBoost)
#modelo_XGB = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=6)


#% 4.6 Criar a pipeline com o pré-processador e o modelo
XGB = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=6))
                             ]) #inseri os detalhes do modelo diretamente no pipeline, para depois poder fazer o GridSearch


#% 4.7 Fit do modelo
XGB.fit(x_train, y_train)


#% 4.8 Ver os dados transformados (pelo preprocessor)
x_train_transformed = preprocessor.transform(x_train)
trans_train = pd.DataFrame(x_train_transformed) #colocar as colunas na ordem em que foram processadas (primeiro as numéricas e depois as categóricas)
trans_train.head()


#% 4.9 Previsões do modelo
previsões_XGB = XGB.predict(x_test)

#% 5.0 Avaliar o modelo
score_XGB = mean_absolute_error(y_test, previsões_XGB)
print('MAE:', score_XGB)
#0.15083798882681565


#%% O modelo que teve melhor score foi o XGBoost

#%% 5. Cross Validation do modelo escolhido (para evitar o overfitting)

# número de dobras = 5
# número de repetições = 3
num_folds = 5
num_repeats = 3
rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=6)
scores_rkf = cross_val_score(XGB, x_train, y_train, cv=rkf, scoring='accuracy')
print("Precisão para cada fold:", ["%.2f" % score for score in scores_rkf])
print("Precisão média: %.2f" % scores_rkf.mean())
#0.79


#%% 6. Tuning do modelo

# Definir a grade de hiperparâmetros que você deseja pesquisar
param_grid = {
    'model__n_estimators': [100, 200, 300], #nº de árvores
    'model__max_depth': [None, 10, 20, 30], #nº de camadas
    'model__min_samples_split': [2, 5, 10], #nº mínimo de amostras necessárias para dividir um nó interno da árvore
    'model__min_samples_leaf': [1, 2, 4] #nº mínimo de amostras necessárias para que uma folha (nó terminal) seja criada
}

# Vou usar o GridSearchCV para pesquisar os hiperparâmetros
#cv = cross validation a usar (estou a usar a rkf que fiz anteriormente)
grid_search = GridSearchCV(estimator=XGB, param_grid=param_grid, cv=rkf, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)
melhores_hiperparametros_gs = grid_search.best_params_

# Obter o melhor modelo encontrado pelo GridSearch
melhor_modelo_gs = grid_search.best_estimator_

# Avaliar o melhor modelo nos dados de teste
acuracia_teste = melhor_modelo_gs.score(x_test, y_test)

# Obtenha a melhor pontuação (score)
melhor_pontuacao_gs = grid_search.best_score_

# Exibir os melhores hiperparâmetros e a acurácia no conjunto de teste
print("Melhores Hiperparâmetros:", melhores_hiperparametros_gs)
print("Melhor Estimador (Modelo):", melhor_modelo_gs.get_params())
print("Melhor Pontuação (Score): %.2f" % melhor_pontuacao_gs) #0.80
print("Acurácia no Conjunto de Teste: %.2f" % acuracia_teste) #0.85


#%% 7. Treinamento do modelo de XGBoost (tuned)

XGB_GS = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', XGBClassifier(
                                  n_estimators=melhores_hiperparametros_gs['model__n_estimators'],
                                  max_depth=melhores_hiperparametros_gs['model__max_depth'],
                                  min_samples_split=melhores_hiperparametros_gs['model__min_samples_split'],
                                  min_samples_leaf=melhores_hiperparametros_gs['model__min_samples_leaf'],
                                  random_state=6  # Defina a semente aleatória se necessário
                              ))
                             ])

XGB_GS.fit(x_train, y_train)

previsões_XGB_GS = XGB_GS.predict(x_test)

score_XGB_GS = mean_absolute_error(y_test, previsões_XGB_GS)
print('MAE:', score_XGB_GS)
#0.16201117318435754
#pior resultado do que do modelo inicial (XGB)


#%% 8. Medição da precisão do modelo após o ajuste

accuracy_GS = accuracy_score(y_test, previsões_XGB)
print(f'Precisão do modelo (Grid Search): {accuracy_GS:.2f}')
#0.85

#%% 9. Curva ROC do Grid Search
y_probs = XGB.predict_proba(x_test)

# Calcular as taxas de verdadeiro positivo (TPR) e as taxas de falso positivo (FPR) usando as probabilidades previstas e os rótulos reais
fpr, tpr, thresholds = roc_curve(y_test, y_probs[:, 1])

# Calcular a área sob a curva ROC (AUC-ROC)
auc_roc = roc_auc_score(y_test, y_probs[:, 1])

# Plot da curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Valor da AUC-ROC
print("AUC-ROC: %.2f" % auc_roc)
#0.87


#%% 10. Matriz de Confusão

# 'y_test' são as classes reais e 'previsões_XGB_GS' são as previsões do modelo
conf_matrix = confusion_matrix(y_test, previsões_XGB)
print("Matriz de Confusão:")
print(conf_matrix)
#TP: 108  |  FN: 5
#FP: 22   |  TN: 44


#%% 11. Fazer previsões no dataset de test

previsões_finais = XGB.predict(test)


#%% 12. Guardar as previsões em ficheiro

Titanic = pd.DataFrame({'PassengerId': test.PassengerId,
                        'Survived': previsões_finais})
Titanic.to_csv('G:/O meu disco/Formação/Kaggle/Kaggle Titanic/Outputs/Titanic.csv', index=False)