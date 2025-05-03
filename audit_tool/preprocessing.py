# audit_tool/preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OrdinalEncoder


def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def overview_data(df):
    """Show basic information and basic statistics of the dataset."""
    print("\nData Overview")
    print("-" * 40)
    print(f"Shape: {df.shape}")
    print("\nDataset Info:")
    print("-" * 40)
    print(df.info())
    print("\nStatistical Summary:")
    print("-" * 40)
    display(df.describe())

def check_missing_values(df, show_only=True, threshold=None):
    """
    Analyze missing values.
    Parameters:
    - show_only: if True, returns only columns with missing values.
    - threshold: if provided, filters columns with percentage of missing values above this threshold.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_percent = (missing / len(df)) * 100
    missing_table = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percent}).sort_values(by='Percentage', ascending=False)

    if threshold:
        missing_table = missing_table[missing_table['Percentage'] > threshold]
    
    if show_only:
        return missing_table
    else:
        return missing_table
    
def impute_missing_values(df, num_method='median', cat_method='constant', cat_fill='Missing'):
    """Impute missing values separately for numerical and categorical columns.

    Args:
        df (pd.DataFrame): DataFrame to process.
        num_method (str): 'mean' or 'median' for numerical columns.
        cat_method (str): 'constant' or 'mode' for categorical columns.
        cat_fill (str): Value to fill if cat_method is 'constant'.
    """
    df_imputed = df.copy()
    num_cols, cat_cols = get_numerical_and_categorical_columns(df)

    # Numerical columns
    if num_method == 'median':
        for col in num_cols:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
    elif num_method == 'mean':
        for col in num_cols:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
    else:
        raise ValueError("Invalid num_method. Choose 'mean' or 'median'.")

    # Categorical columns
    if cat_method == 'constant':
        for col in cat_cols:
            df_imputed[col] = df_imputed[col].fillna(cat_fill)
    elif cat_method == 'mode':
        for col in cat_cols:
            if not df_imputed[col].mode().empty:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
    else:
        raise ValueError("Invalid cat_method. Choose 'constant' or 'mode'.")

    return df_imputed

def detect_outliers(df, threshold=1.5):
    """Detect numerical outliers using the IQR method."""
    outlier_summary = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - threshold * IQR) | (df[col] > Q3 + threshold * IQR)]
        if not outliers.empty:
            outlier_summary[col] = len(outliers)
    return dict(sorted(outlier_summary.items(), key=lambda item: item[1], reverse=True))


def plot_variable_distribution(df, column, bins=50):
    """Plot histogram of a variable."""
    plt.figure(figsize=(10, 4))
    plt.hist(df[column].dropna(), bins=bins, edgecolor='k')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_multiple_distributions(df, columns, bins=50, n_cols=3, figsize=(15, 10)):
    """Plot multiple variable distributions in a grid."""
    n_rows = (len(columns) + n_cols - 1) // n_cols  # calcula filas necesarias
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        axes[i].hist(df[col].dropna(), bins=bins, edgecolor='k')
        axes[i].set_title(col)
        axes[i].grid(True)

    # Quitar subplots vacíos si sobran
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def calculate_outlier_percentage(df, threshold=1.5):
    """Calculate outlier percentage per column."""
    outlier_summary = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_rows = df.shape[0]
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - threshold * IQR) | (df[col] > Q3 + threshold * IQR)]
        percentage = len(outliers) / n_rows * 100
        if percentage > 0:
            outlier_summary[col] = percentage
    return dict(sorted(outlier_summary.items(), key=lambda item: item[1], reverse=True))


def treat_outliers(df, method='capping', threshold=1.5):
    """
    Treat outliers in the dataframe.
    method: 'capping' (default) to clip extreme values to IQR limits, or 'removal' to drop rows with outliers.
    """
    df_copy = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        if method == 'capping':
            df_copy[col] = np.where(df_copy[col] < lower_bound, lower_bound,
                             np.where(df_copy[col] > upper_bound, upper_bound, df_copy[col]))
        
        elif method == 'removal':
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
        
    return df_copy

def winsorize_column(df, column, lower_quantile=0.0, upper_quantile=0.99):
    """Apply winsorization to a single column."""
    lower = df[column].quantile(lower_quantile)
    upper = df[column].quantile(upper_quantile)
    df[column] = df[column].clip(lower, upper)
    return df

def winsorize_columns(df, columns, limits=(0, 0.01)):
    """Apply winsorization to specified columns."""
    df_winsorized = df.copy()
    for col in columns:
        lower_lim, upper_lim = limits
        df_winsorized[col] = winsorize(df[col], limits=(lower_lim, upper_lim))
    return df_winsorized


def encode_categoricals(df, encoding_type='onehot'):
    """
    Encode categorical variables.
    encoding_type: 'onehot' for One-Hot Encoding (default), 'ordinal' for Ordinal Encoding.
    """

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if encoding_type == 'onehot':
        return pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    elif encoding_type == 'ordinal':
        df_copy = df.copy()
        encoder = OrdinalEncoder()
        df_copy[cat_cols] = encoder.fit_transform(df_copy[cat_cols])
        return df_copy
    
    else:
        raise ValueError("encoding_type must be 'onehot' or 'ordinal'")
    

def encode_mixed_categoricals(df, ordinal_vars, onehot_vars):
    """
    Apply ordinal encoding to ordinal_vars and one-hot encoding to onehot_vars.
    """
    df_encoded = df.copy()

    # Ordinal encoding
    if ordinal_vars:
        encoder = OrdinalEncoder()
        df_encoded[ordinal_vars] = encoder.fit_transform(df_encoded[ordinal_vars])

    # One-hot encoding
    df_encoded = pd.get_dummies(df_encoded, columns=onehot_vars, drop_first=True)

    return df_encoded

def safe_extract_date_features(df, date_cols):
    """
    Safely extract year and month from date columns if possible.
    If a column cannot be converted to datetime, it will be skipped.
    """
    df = df.copy()
    for col in date_cols:
        try:
            # Try converting to datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if df[col].notnull().any():
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                print(f"Processed date column: {col}")
            else:
                print(f"Column '{col}' has no valid dates. Skipping.")
        except Exception as e:
            print(f"Could not process column '{col}': {e}")

    return df


def scale_features(df, target_column=None):
    """
    Scale numerical features using StandardScaler.
    If target_column is specified, it will be excluded from scaling.
    """
    from sklearn.preprocessing import StandardScaler

    df_scaled = df.copy()

    if target_column and target_column in df_scaled.columns:
        y = df_scaled[target_column]
        X = df_scaled.drop(columns=[target_column])
    else:
        X = df_scaled
        y = None

    numeric_cols = X.select_dtypes(include=[np.number]).columns

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    if y is not None:
        X[target_column] = y  # Reattach target unscaled

    return X


def save_processed_data(df, path):
    """Save processed dataset to a CSV file."""
    df.to_csv(path, index=False)

def get_numerical_and_categorical_columns(df):
    """Return lists of numerical and categorical columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return num_cols, cat_cols
