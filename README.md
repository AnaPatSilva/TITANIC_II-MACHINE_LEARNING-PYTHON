![titanic](https://github.com/AnaPatSilva/Titanic_II-Machine-Learning-Python/blob/main/titanic_custom-fc6a03aedd8e562d780ecf9b9a8a947d4dcbf163-s1100-c50.jpg)
# Titanic_II
I made my first submission to the Kaggle Competition [**Titanic - Machine Learning from Disaster**](https://www.kaggle.com/competitions/titanic/overview).

You can check it [here](https://github.com/AnaPatSilva/Titanic-I).

The score was 0.77511 but I want to have a perfect score (1), so I keep trying to improve my model.

This is my second model/submission.

## New submission
[Code (Python): New Submission](https://github.com/AnaPatSilva/Titanic-II/blob/main/Code%20(Python)/Titanic.py)

In this submission I used the same variables ('Sex', 'Pclass', 'Fare'), tried different models (Random Forest Regression, Random Forest Classifier, and XGBoost) and used pipelines.

**Steps:**
1. **Importing Libraries and Data:**
- The necessary libraries and modules are imported.
- Titanic dataset files (gender_submission.csv, test.csv, and train.csv) are read into Pandas DataFrames.
2. **Exploratory Data Analysis (EDA):**
- Data profiling using the ydata_profiling library ([Profile_Train.pdf](https://github.com/AnaPatSilva/Titanic-II/blob/main/Data%20Profiling/profile_train.pdf)).
- Creation of a correlation matrix and visualization using a heatmap.
- Identification of features most correlated with the 'Survived' target variable.
3. **Data Preprocessing:**
- Checking for missing values in selected variables ('Survived', 'Sex', 'Pclass', 'Fare').
- Splitting the dataset into training and testing sets.
4. **Machine Learning Models:**
- Attempted various models including Random Forest Regression, Random Forest Classifier, and XGBoost.
- Different preprocessing approaches, including using pipelines and transformers.
- Evaluation of models using mean absolute error (MAE) and cross-validation.
- Model tuning using GridSearchCV.
5. **Model Evaluation and Output:**
- Evaluation of the tuned XGBoost model.
- Generation of a ROC curve and calculation of AUC-ROC.
- Creation of a confusion matrix.
- Making predictions on the test dataset and saving the results to a CSV file ([Titanic.csv](https://github.com/AnaPatSilva/Titanic-II/blob/main/Outputs/Titanic.csv)).

## Tests
While doing this model I tried several approaches. I’ve written them down in the main code, but I’ll also leave them here individually.
1. [**Random Forest Regression with pipeline and categorial and numeric transformers**](https://github.com/AnaPatSilva/Titanic-II/blob/main/Code%20(Python)/teste.py)
2. [**Random Forest Regression with pipeline and categorial (if binary) and numeric transformers**](https://github.com/AnaPatSilva/Titanic-II/blob/main/Code%20(Python)/teste1.py)
3. [**Random Forest Regression with pipeline and categorial transformers**](https://github.com/AnaPatSilva/Titanic-II/blob/main/Code%20(Python)/teste2.py)
4. [**Random Forest Classifier without pipeline and with categorial one hot coding**](https://github.com/AnaPatSilva/Titanic-II/blob/main/Code%20(Python)/teste3.py)
5. [**Random Forest Classifier with pipeline and categorial and numeric transformers**](https://github.com/AnaPatSilva/Titanic-II/blob/main/Code%20(Python)/teste4.py)
6. [**Random Forest Classifier with pipeline and categorial (if binary) and numeric transformers**](https://github.com/AnaPatSilva/Titanic-II/blob/main/Code%20(Python)/teste5.py)
7. [**Random Forest Classifier with pipeline and categorial transformers**](https://github.com/AnaPatSilva/Titanic-II/blob/main/Code%20(Python)/teste6.py)

## Conclusions
The model with the best score was XGBoost, but my score in the competition was **0.76555** (worst than my first submission ☹).

Next time, I will try other approaches. Feel free to give me some tips or guidelines.
