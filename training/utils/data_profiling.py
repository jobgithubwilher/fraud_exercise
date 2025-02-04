import pandas as pd
import numpy as np

def analyze_dataframe(df):
    """
    Analyzes the provided DataFrame, calculating statistical summaries for numerical
    and categorical columns, as well as time-related columns. It computes:
    - Type of variable (numerical, categorical, or time).
    - Basic statistics: min, max, mean, median, Q1, Q3, and number of missing values.
    - If categorical, the number of unique categories and the frequency of each category (absolute and percentage).
    - If time-related, min and max values.
    - For boolean columns, unique values, their frequencies (absolute and percentage).
    
    Parameters:
    - df: pandas DataFrame to be analyzed
    
    Returns:
    - A dictionary with column names as keys and analysis results as values
    """
    
    analysis_result = {}
    
    for column in df.columns:
        col_analysis = {}

        # Check if the column is a datetime column
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            col_data = df[column]
            
            # Min and Max of the datetime column
            col_analysis['Min'] = col_data.min()
            col_analysis['Max'] = col_data.max()
        
        # Check for boolean columns
        elif pd.api.types.is_bool_dtype(df[column]):
            col_data = df[column]
            col_analysis['Unique Categories'] = col_data.nunique()
            
            # Frequency and percentage for boolean columns
            freq = col_data.value_counts()
            col_analysis['Category Frequencies'] = {
                k: {"Count": v, "Percentage": f"{round(v / len(col_data) * 100, 2)}%" } for k, v in freq.items()}
            col_analysis['Missing'] = col_data.isnull().sum()
        
        else:
            # For non-time columns, calculate other stats
            col_data = df[column]
            
            # Type of the column (categorical, numerical, etc.)
            col_analysis['Type'] = col_data.dtype
            
            # If it's numerical, calculate statistics
            if np.issubdtype(col_data.dtype, np.number):
                col_analysis['Min'] = col_data.min()
                col_analysis['Max'] = col_data.max()
                col_analysis['Mean'] = col_data.mean()
                col_analysis['Median'] = col_data.median()
                col_analysis['Q1'] = col_data.quantile(0.25)
                col_analysis['Q3'] = col_data.quantile(0.75)
                col_analysis['Missing'] = col_data.isnull().sum()
            
            # If it's categorical, calculate unique categories and their frequencies
            elif col_data.dtype == 'object' or pd.api.types.is_categorical_dtype(col_data):
                col_analysis['Unique Categories'] = col_data.nunique()
                
                # Frequency and percentage for categorical columns
                if col_data.nunique() <= 10:  # If less than or equal to 10 unique categories
                    freq = col_data.value_counts()
                    col_analysis['Category Frequencies'] = {
                        k: {"Count": v, "Percentage": f"{round(v / len(col_data) * 100, 2)}%" } for k, v in freq.items()}
                col_analysis['Missing'] = col_data.isnull().sum()

        # Store the analysis for each column
        analysis_result[column] = col_analysis
    
    return analysis_result
