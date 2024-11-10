# Data manipulation
import numpy as np
import pandas as pd

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# File handling
import os

# Creating a new respository to save study plots 
PATH_Plots = './plots_saved'
if not os.path.exists(PATH_Plots):
    os.makedirs(PATH_Plots)
    
# Import train and test data
test = pd.read_csv('./test.csv', index_col=0)
train = pd.read_csv('./train.csv', index_col=0)
print("Data Imported !")

# Using abs values for negative values
train['Vertical_Distance_To_Hydrology'] = abs(train['Vertical_Distance_To_Hydrology'])
test['Vertical_Distance_To_Hydrology'] = abs(test['Vertical_Distance_To_Hydrology'])

# Setting seed for reproductibility
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
                   check_histogram_continuous_variables=False):
    """
    This function provides various analyses and visualizations on the dataset, 
    including correlation matrix, boxplots, occurrence of binary variables, 
    and histograms for continuous variables. You can enable specific checks 
    using the respective parameters.

    Parameters:
    - check_corr_matrix: Boolean to check and plot the correlation matrix of selected variables.
    - check_box_plot: Boolean to create boxplots for continuous variables grouped by 'Cover_Type'.
    - check_histogram_continuous_variables: Boolean to plot histograms of continuous variables for train and test datasets.
    """
    
    # Select relevant continuous variables from the dataset
    variables = ['Elevation', 'Aspect', 'Slope', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Hydrology',
                'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Hillshade_9am', 
                'Hillshade_Noon', 'Hillshade_3pm']
        
    if check_corr_matrix == True :

        # Compute the correlation matrix
        correlation_matrix = train[variables].corr()

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
        top_correlations.to_csv(f"{PATH_Plots}/top_abs_correlated_features.csv", index=True)
        
    if check_box_plot == True :
        
        # Count plot for Cover_Type
        plt.figure(figsize=(10, 6))
        sns.countplot(data=train, x='Cover_Type')
        plt.title('Cover Type Distribution')
        plt.xlabel('Cover Type')
        plt.ylabel('Count')
        plt.savefig(f"{PATH_Plots}/cover_type_countplot.jpg", format="jpeg")
        plt.close()
        
        # Wilderness_Area distribution by Cover_Type
        wilderness_cols = [col for col in train.columns if 'Wilderness_Area' in col]
        wilderness_melted = train.melt(id_vars=['Cover_Type'],
                                       value_vars=wilderness_cols,
                                       var_name='Wilderness_Area',
                                       value_name='Presence')
        wilderness_present = wilderness_melted[wilderness_melted['Presence'] == 1]

        plt.figure(figsize=(12, 6))
        sns.countplot(x='Wilderness_Area', hue='Cover_Type', data=wilderness_present)
        plt.title('Wilderness Area vs Forest Cover Type')
        plt.xticks(rotation=45)
        plt.legend(title='Cover Type')
        plt.tight_layout()
        plt.savefig(f"{PATH_Plots}/wilderness_area_vs_cover_type.jpeg")
        plt.close()

        # Soil_Type distribution by Cover_Type
        soil_type_cols = [col for col in train.columns if 'Soil_Type' in col]
        soil_cover_type = train.groupby('Cover_Type')[soil_type_cols].sum().T

        plt.figure(figsize=(15, 10))
        soil_cover_type.plot(kind='bar', stacked=True, figsize=(15, 8), colormap='Set1')
        plt.title('Distribution of Cover_Type across Soil_Type', fontsize=16)
        plt.xlabel('Soil_Type', fontsize=12)
        plt.ylabel('Count of Cover_Type', fontsize=12)
        plt.legend(title="Cover_Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{PATH_Plots}/soil_type_vs_cover_type.jpeg")
        plt.close()
        
        # Adjust color palette to match the number of unique Cover_Type values
        num_categories = train['Cover_Type'].nunique()
        cmap = sns.color_palette("Set1", num_categories)

        for label in variables:
            plt.figure(figsize=(plt.gcf().get_size_inches()))
            sns.boxplot(x='Cover_Type', y=label, data=train, palette=cmap, hue='Cover_Type', dodge=False)
            plt.legend([], [], frameon=False)  # Remove legend to avoid redundant information
            plt.savefig(f"{PATH_Plots}/boxplot_{label}.jpeg")
            plt.close()
        
    if check_histogram_continuous_variables == True :
        
        # Plot histogram for the train dataset and save it as a JPEG
        train.iloc[:, 0:9].hist(figsize=(10, 8), bins=50)
        plt.savefig(f"{PATH_Plots}/train_histogram.jpg", format="jpeg")
        plt.close()

        # Plot histogram for the test dataset and save it as a JPEG
        test.iloc[:, 0:9].hist(figsize=(10, 8), bins=50)
        plt.savefig(f"{PATH_Plots}/test_histogram.jpg", format="jpeg")
        plt.close()
        
# Main execution block
if __name__ == "__main__":
    # Set your flags or parameters as needed
    check_duplicates(check=True)  
    drop_missing_values(train, drop=True) 
    data_set_study(check_corr_matrix=True, 
                   check_box_plot=True, 
                   check_histogram_continuous_variables=True)
