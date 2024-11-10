# Data manipulation
import numpy as np
import pandas as pd

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# File handling
import os

#Creating a new respository to save study plots 
PATH_Plots = './plots_saved'
if not os.path.exists(PATH_Plots):
    os.makedirs(PATH_Plots)

#Creating a new respository to store new dfs
PATH_DFs = './new_dfs'
if not os.path.exists(PATH_DFs):
    os.makedirs(PATH_DFs)
    
#Import train and test data
test = pd.read_csv('./test.csv', index_col=0)
train = pd.read_csv('./train.csv', index_col=0)
print("Data Imported !")

#Setting seed for reproductibility
seed = 42

def check_duplicates(check=False):
    
    """
    This function checks for duplicate rows in the training dataset and 
    prints the count of duplicates.
    """
    if check == True:
        
        dupplicates_count = train.duplicated().sum()
        
    print(f"There's {dupplicates_count} duplicates")

def drop_missing_values(data, drop=False):
    """
    This function checks for missing values in the training dataset by 
    counting the number of missing values per column and then drops rows 
    with any missing values from the dataset.
    """
    missing_values_count = train.isnull().sum()
    print("Missing values per column:")
    print(missing_values_count)
    
    if drop == True:
        
        # Drop rows with any missing values
        data.dropna(inplace=True)
        
def data_set_study(check_corr_matrix=False, 
                   check_box_plot=False,
                   check_histogram=False):
    """
    This function provides various analyses and visualizations on the dataset, 
    including correlation matrix, boxplots, occurrence of binary variables, 
    and histograms for continuous variables. You can enable specific checks 
    using the respective parameters.

    Parameters:
    - check_corr_matrix: Boolean to check and plot the correlation matrix of selected variables.
    - check_box_plot: Boolean to create boxplots for continuous variables grouped by 'Cover_Type'.
    - check_occurence_binary_variable: Boolean to analyze and plot the occurrence of binary soil types for each cover type.
    - check_histogram_continuous_variables: Boolean to plot histograms of continuous variables for train and test datasets.
    """
        
    if check_corr_matrix == True :

        numerical_cols = train.select_dtypes(include='number').columns.tolist()

        # Compute the correlation matrix
        correlation_matrix = train[numerical_cols].corr()

        # Plot and save the correlation matrix
        plt.figure(figsize=(16, 12))
        mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=-1)
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, mask=mask)
        plt.xticks(rotation=65, ha='right')
        plt.yticks(rotation=0)
        plt.title("Correlation Matrix of Selected Variables")
        plt.savefig(f"{PATH_Plots}/corr_matrix.jpeg")
        plt.close()

        # Compute the absolute correlation matrix
        abs_correlation_matrix = correlation_matrix.abs()

        # Get the upper triangle of the absolute correlation matrix
        upper_triangle_indices = np.triu_indices_from(abs_correlation_matrix, k=1)
        abs_correlation_values = abs_correlation_matrix.values[upper_triangle_indices]
        correlation_values = correlation_matrix.values[upper_triangle_indices]
        feature_pairs = [(correlation_matrix.columns[i], correlation_matrix.columns[j]) 
                        for i, j in zip(*upper_triangle_indices)]

        # Create a DataFrame for the correlations
        correlation_pairs = pd.DataFrame({
            'Feature Pair': feature_pairs,
            'Absolute Correlation': abs_correlation_values,
            'Correlation Coefficient': correlation_values
        })

        # Sort the DataFrame by absolute correlation values
        top_correlations = correlation_pairs.sort_values(by='Absolute Correlation', ascending=False).head(20)

        # Save the DataFrame to a CSV file for better readability
        top_correlations.to_csv(f"{PATH_DFs}/top_abs_correlated_features.csv", index=True)
        
    if check_box_plot == True :
        # RatecodeID Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='RatecodeID', y='tip_amount', data=train)
        plt.title('Tip Amount Distribution by Ratecode ID')
        plt.xlabel('Ratecode ID')
        plt.ylabel('Tip Amount')
        plt.savefig(f"{PATH_Plots}/ratecode_boxplot.jpeg")
        plt.close()
        
        # Payment Type Bar Plot
        plt.figure(figsize=(10, 6))
        payment_counts = train[train['tip_amount'] > 0].groupby('payment_type').size()
        plt.bar(payment_counts.index, payment_counts.values)
        plt.xlabel('Payment Type of tip amount > 0', fontsize=18)
        plt.ylabel('Number of Trips', fontsize=18)
        plt.title("Number of Trips by Payment Type")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"{PATH_Plots}/Payment_Type_Distribution.jpeg")
        plt.close()
                
    if check_histogram == True:
        columns_list = [
            "VendorID", "passenger_count", "payment_type", 
            "tip_amount", "tolls_amount", "trip_distance_miles"
        ]
        
        for column in columns_list:
            plt.figure(figsize=(10, 6))  # Set figure size for clarity

            # Plot the histogram
            ax = sns.countplot(x=column, data=train)
            plt.title(f'Distribution of {column}')
            
            # Adjust x-axis labels for large datasets
            max_value = train[column].max() if train[column].dtype != 'object' else len(train[column].unique())
            
            if max_value > 10:  # Assuming "large" dataset if there are more than 10 unique values
                print(f"For {column}, max value is {max_value}")
                
                # Calculate the interval for labels to appear only at quarter positions
                ticks_interval = max(1, int(max_value) // 4)
                labels_of_interest = [str(i) for i in range(0, int(max_value) + 1, ticks_interval)]
                
                # Get all current tick labels
                original_labels = [str(label) for label in ax.get_xticks()]
                # Set labels only for quarters, blank for others
                new_labels = [label if label in labels_of_interest else "" for label in original_labels]
                
                # Apply the new labels to the x-axis
                ax.set_xticklabels(new_labels)
                
            # Save the plot to a file
            plt.savefig(f"{PATH_Plots}/histogram_{column}.jpeg")
            plt.close()
    
    