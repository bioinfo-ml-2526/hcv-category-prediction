import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def  exploratory_data_analysis(file_path, biomarkers):
    # opening the file
    df = pd.read_csv(file_path, sep=',', names=['id', 'category', 'age', 'sex', 'alb', 'alp', 'alt', 'ast', 'bil', 'che', 'chol', 'crea', 'cgt', 'prot'])
    
    # droping the first row
    df.drop(df.index[0], inplace=True)
    
    print('\n--------------------------------------------- EXPLORATORY DATA ANALYSIS ---------------------------------------------\n')
    print(f'# Observations and features:\n{df.shape}')
    print(f'\n# Top 5 observations:\n{df.head()}')
    print(f'\n# Last 5 observations:\n{df.tail()}')
    print(f'\n# Info about dataframe:')
    df.info()
    print(f'\n# Number of unique elements:\n{df.nunique()}')
    print(f'\n# Missing values: \n {df.isnull().sum()}')
    print(f'\n# Duplicated values: {df.duplicated().sum()}')
    print(f'\n# Statistics:\n{df.describe(include='all').T}')
    print('\n--------------------------------------------- EXPLORATORY DATA ANALYSIS ---------------------------------------------\n')
        
    return df



def encoding(df):
    # one-hot enconding for sex and category columns
    df = pd.get_dummies(df, columns=['sex'], prefix='sex', drop_first=True, dtype=int)
    df = pd.get_dummies(df, columns=['category'], prefix='cat', drop_first=True, dtype=int)
    
    return df



def missing_values(df):
    # filling missing values
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
        max_iter=10,
        random_state=42
    )
    
    array = imputer.fit_transform(df)
    df = pd.DataFrame(array, columns=df.columns)
    
    print(f'\n# Missing values after imputation: \n {df.isnull().sum()}')
    
    return df



def cleaning_outliers(df, biomarkers):
    # correcting outliers
    for bm in biomarkers:    
        print(f'\n---------------------------------------------')
        print(f'\n# Metrics {bm.upper()} before cleaning outliers: \n {df[bm].describe()}')
        # plt.boxplot(df[bm])
        # plt.show()
        # all biomarkers has outliers
        
        q1 = df[bm].quantile(0.25)
        q3 = df[bm].quantile(0.75)
        iqr = q3 - q1
        
        infw = q1 - 1.5 * iqr
        supw = q3 + 1.5 * iqr
        
        median = df[bm].median()
        
        df[bm] = np.where(
            (df[bm] < infw) | (df[bm] > supw),
            median,
            df[bm]
        )
        print(f'\n# Metrics {bm.upper()} after cleaning outliers: \n {df[bm].describe()}')
        # plt.boxplot(df[bm])
        # plt.show()
        
    return df



def normalizing(df, numeric_col):
    scaler = StandardScaler()
    numeric_col.append('age')
    
    df[numeric_col] = scaler.fit_transform(df[numeric_col])
    
    return df



def pre_processing(df, biomarkers):
    print('\n--------------------------------------------- DATA PRE-PROCESSING ---------------------------------------------')
    
    print('\n> Enconding categorical variables...')
    df = encoding(df)
    
    print('\n> Filling missing values...')
    df = missing_values(df)
    
    print('\n> Correcting outliers...')
    df = cleaning_outliers(df, biomarkers)
    
    print('\n> Normalizing data...')
    df = normalizing(df, biomarkers)
    
    print(f'\n> Statistics:\n{df.describe(include='all').T}')
    print('\n--------------------------------------------- DATA PRE-PROCESSING ---------------------------------------------')
    
    return df
    
    
    
if __name__ == '__main__':
    file_path = 'hcvdata/hcvdat0.csv'
    biomarkers = ['alb', 'alp', 'alt', 'ast', 'bil', 'che', 'chol', 'crea', 'cgt', 'prot']
    initial_dataframe = exploratory_data_analysis(file_path, biomarkers)
    cleaned_dataframe = pre_processing(initial_dataframe, biomarkers)