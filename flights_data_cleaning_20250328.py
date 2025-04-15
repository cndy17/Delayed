#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning

# Create a separate Notebook that will clean your data by removing records with missing data, filling in missing values where appropriate, dropping unneeded columns and applying appropriate data types to all columns. Be sure to rename columns and remove spaces from column names. The Data Cleaning code will read data from /landing folder, apply the schema to the data, fill in nulls or remove records with nulls, remove unnecessary columns and then write the data to the /cleaned folder as a Parquet file.

# In[1]:


# Import the storage module
from google.cloud import storage
from io import StringIO, BytesIO
import pandas as pd
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.width', 1000)


# In[2]:


def perform_EDA(df : pd.DataFrame, filename : str):
    """
    perform_EDA(df : pd.DataFrame, filename : str)
    Accepts a dataframe and a text filename as inputs.
    Runs some basic statistics on the data and outputs to console.

    :param df: The Pandas dataframe to explore
    :param filename: The name of the data file
    :returns:
    """
    print(f"{filename} Number of records:")
    print(df.count())
    number_of_duplicate_records = df.duplicated().sum()
    # old way number_of_duplicate_records = len(df)-len(df.drop_duplicates())
    print(f"{filename} Number of duplicate records: {number_of_duplicate_records}" )
    print(f"{filename} Info")
    print(df.info())
    print(f"{filename} Describe")
    print(df.describe())
    print(f"{filename} Columns with null values")
    print(df.columns[df.isnull().any()].tolist())
    rows_with_null_values = df.isnull().any(axis=1).sum()
    print(f"{filename} Number of Rows with null values: {rows_with_null_values}" )
    integer_column_list = df.select_dtypes(include='int64').columns
    print(f"{filename} Integer data type columns: {integer_column_list}")
    float_column_list = df.select_dtypes(include='float64').columns
    print(f"{filename} Float data type columns: {float_column_list}")



# In[3]:


def perform_EDA_numeric(df : pd.DataFrame, filename : str):
    """
    perform_EDA_numeric(df : pd.DataFrame, filename : str)
    Accepts a dataframe and a text filename as inputs.
    Runs some basic statistics on numeric columns and saves the output in a dataframe.

    :param df: The Pandas dataframe to explore
    :param filename: The name of the data file
    :returns:
    :   pd.DataFrame: A new dataframe with summary statistics
    """
    # Initialize a list to collect summary data
    summary_data = []
    # Gather summary statistics on numeric columns
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
       summary_data.append({
            'Filename': filename,         'Column': col,
            'Minimum': df[col].min(),     'Maximum': df[col].max(),
            'Average': df[col].mean(),    'Standard Deviation': df[col].std(),
            'Missing Values': df[col].isnull().sum()
        })
    # Convert the summary data list into a DataFrame
    return pd.DataFrame(summary_data)



# In[4]:


def perform_EDA_categorical(df : pd.DataFrame, filename : str, categorical_columns):
    """
    perform_EDA_categorical(df : pd.DataFrame, filename : str, categorical_columns)
    Accepts a dataframe and a text filename as inputs.
    Collects statistics on Categorical columns

    :param df: The Pandas dataframe to explore
    :param filename: The name of the data file
    :param categorical_columns: A list of column names for categorical columns
    :returns:
    :   pd.DataFrame: A new dataframe with summary statistics
    """
    # Initialize a list to collect summary data
    summary_data = []
    # Gather summary statistics on numeric columns
    for col in categorical_columns:
       summary_data.append({
            'Filename': filename,
            'Column': col,
            'Unique Values': df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x).nunique(),
            'Minimum': df[col].min(),
            'Maximum': df[col].max(),
            'Missing Values': df[col].isnull().sum()
        })
    # Convert the summary data list into a DataFrame
    return pd.DataFrame(summary_data)


# In[5]:


def plot_categorical_distributions(df, categorical_columns):
    for col in categorical_columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=col, data=df)
        plt.title(f"Frequency of Categories for {col}")
        plt.xlabel(col)
        plt.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation = 45)
        plt.ylabel('Count')
        plt.show()


# In[6]:


def plot_numerical_distributions(df, numerical_columns):
    for col in numerical_columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], discrete=True, kde=False, bins=10, color='blue')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation = 45)
        plt.xlabel(col)
        plt.show()


# In[7]:


def make_categorical_visualizations(df : pd.DataFrame, filename : str, categorical_columns):
    """
    make_categorical_visualizations(df : pd.DataFrame, filename : str, categorical_columns)
    Accepts a dataframe, a text filename and a list of categorical columns as inputs.
    Creates count plots on the categorical columns using Seaborn

    :param df: The Pandas dataframe to explore
    :param filename: The name of the data file
    :param categorical_columns: A list of column names for categorical columns
    :returns:
    :   None
    """
    # Loop over all of the categorical columns   
    for col in categorical_columns:
        plt.figure(figsize=(16, 8)) 
        sns.countplot(y=col, data=df)  # Horizontal bars
        plt.title(f"{filename} Frequency of Categories for {col}")
        plt.ylabel(col)
        plt.ticklabel_format(style='plain', axis='x') 
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels and align them to the right
        plt.xlabel('Count')
        plt.tight_layout() 
        plt.show()


# In[8]:


def make_continuous_visualizations(df : pd.DataFrame, filename : str, continuous_columns):
    """
    make_continuous_visualizations(df : pd.DataFrame, filename : str, continuous_columns)
    Accepts a dataframe, a text filename and a list of continuous columns as inputs.
    Creates  plots using Seaborn

    :param df: The Pandas dataframe to explore
    :param filename: The name of the data file
    :param continuous_columns: A list of column names for categorical columns
    :returns:
    :   None
    """
    for col in continuous_columns:
        plt.figure(figsize=(16, 8))
        # Create a scatter plot for the column (col)
        sns.scatterplot(x=continuous_columns[0], y=col, data=df)
        # Format y-axis to avoid scientific notation
        plt.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation = 45)
        # Set titles, x labels and y labels
        plt.title(f"{filename} Scatter Plot for {continuous_columns[0]} vs. {continuous_columns[1]}")
        plt.xlabel(continuous_columns[0])
        plt.ylabel(col)
        plt.tight_layout() 
        plt.show()


# In[9]:


def filter_relevant_columns(df):
    columns_relevant = ['FlightDate', "Airline", "Origin", "Dest", "Cancelled", "Diverted", 
                                'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek',
                               "Operating_Airline", "OriginAirportID", "OriginCityName", "OriginState", "OriginStateName",
                               "DestAirportID",  "DestCityName", "DestState","DestStateName", 
                               'DepDel15', 'DepartureDelayGroups', 'DepTimeBlk', 'ArrivalDelayGroups',
                                'ArrTimeBlk', 'ArrDel15', 'DistanceGroup','CRSDepTime', 'DepTime', 'DepDelayMinutes', 'DepDelay', 'ArrTime', 'ArrDelayMinutes', 'AirTime', 
                              'CRSElapsedTime', 'ActualElapsedTime', 'Distance', 'TaxiOut', 'WheelsOff', 
                              'WheelsOn', 'TaxiIn', 'CRSArrTime', 'ArrDelay', 'DivAirportLandings']
    df_filtered = df[columns_relevant]
    return df_filtered


# In[10]:


def main_flight():
    # This function processes the flight dataset.
    categorical_columns_list = ['FlightDate', "Airline", "Origin", "Dest", "Cancelled", "Diverted", 
                                'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek',
                               "Operating_Airline", "OriginAirportID", "OriginCityName", "OriginState", "OriginStateName",
                               "DestAirportID",  "DestCityName", "DestState", "DestStateName", 
                               'DepDel15', 'DepartureDelayGroups', 'DepTimeBlk', 'ArrivalDelayGroups',
                                'ArrTimeBlk', 'ArrDel15', 'DistanceGroup']
    numerical_columns_list = ['ActualElapsedTime','CRSDepTime', 'DepTime', 'DepDelayMinutes', 'DepDelay', 'ArrTime', 'ArrDelayMinutes', 'AirTime', 
                              'CRSElapsedTime',  'Distance', 'TaxiOut', 'WheelsOff', 
                              'WheelsOn', 'TaxiIn', 'CRSArrTime', 'ArrDelay', 'DivAirportLandings']

    bucket_name = 'my-project-bucket-flights-cl'
    landing_folder = f"{bucket_name}/landing/"
    cleaned_folder = f"{bucket_name}/cleaned/"

    storage_client = storage.Client()
    
    # Point to the bucket
    bucket = storage_client.get_bucket(bucket_name)

    # Create a client object that points to GCS
    storage_client = storage.Client()
    
    # Get a list of the 'blobs' (objects or files) in the bucket
    blobs = storage_client.list_blobs(bucket_name, prefix="landing/")
    
    # Iterate through the list and print out their names
    parquet_blobs = [blob for blob in blobs if blob.name.endswith('.parquet')]
    for blob in parquet_blobs:
        print(f"file {blob.name} with size {blob.size} bytes created on {blob.time_created}")
        
        # Read in the Parquet file from the blob.
        df = pd.read_parquet(BytesIO(blob.download_as_bytes()))
        print("Columns in the dataset:", df.columns)  # Debugging step
        
        # Apply filtering relevant columns
        df = filter_relevant_columns(df)
        
        # Perform Data Cleaning
        # Remove NAs
        df.dropna(inplace=True)
        
        # Remove outliers
        # Remove Negative Elapsed Time outlier
        df = df[df['ActualElapsedTime'] >= 0]

        # Change to appropriate datatype
        df['DepDel15'] = df['DepDel15'].astype(bool)
        df['ArrDel15'] = df['ArrDel15'].astype(bool)
        # df['DepartureDelayGroups'] = df['DepartureDelayGroups'].astype(int)
        # df['ArrivalDelayGroups'] = df['ArrivalDelayGroups'].astype(int)
        df['OriginAirportID'] = df['OriginAirportID'].astype(str)
        df['DestAirportID'] = df['DestAirportID'].astype(str)
        df = df.astype({col: str for col in df.select_dtypes(include=[object]).columns})
       
        print(f"Schema for {blob.name}")
        df.info()

        # Gather the statistics on numeric columns
        numeric_summary_df = perform_EDA_numeric(df, blob.name)
        print(numeric_summary_df.head(12))
        # Gather statistics on the categorical columns
        categorical_summary_df = perform_EDA_categorical(df, blob.name, categorical_columns_list)
        print(categorical_summary_df.head(12))

        # Create some visualizations for the categorical columns
        categorical_columns = ["Airline", "Cancelled", "Diverted", 
                                'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek',
                               "Operating_Airline", "OriginStateName", "DestStateName", 
                               'DepDel15', 'DepartureDelayGroups', 'DepTimeBlk', 'ArrivalDelayGroups',
                                'ArrTimeBlk', 'ArrDel15', 'DistanceGroup']
        #make_categorical_visualizations(df, blob.name, categorical_columns)
        
        # Create some scatter plots to look for outliers
        # Use a pair of continuous data columns
        continuous_columns = ['ActualElapsedTime', 'DepDelay', 'CRSDepTime', 'DepTime', 'DepDelayMinutes', 'ArrTime', 'ArrDelayMinutes', 'AirTime', 
                              'CRSElapsedTime', 'Distance', 'TaxiOut', 'WheelsOff', 
                              'WheelsOn', 'TaxiIn', 'CRSArrTime', 'ArrDelay', 'DivAirportLandings']

        #make_continuous_visualizations(df, blob.name, continuous_columns)
        
        # Write the data to the /cleaned folder as a Parquet file
        new_filename = blob.name.replace("landing/", "cleaned/").replace(".tsv", ".parquet")
        print(f"Saving cleaned data file to {new_filename}")
        filedata = df.to_parquet(index=False)
        new_blob = bucket.blob(new_filename)
        new_blob.upload_from_string(filedata, content_type='application/octet-stream')

        # For testing one file
        if blob.name == "landing/Combined_Flights_2022.parquet":
            break


# In[11]:


if __name__ == "__main__":
    main_flight()

