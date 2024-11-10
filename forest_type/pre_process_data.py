# Data manipulation
import numpy as np
import pandas as pd

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn models and utilities
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier

# XGBoost
from xgboost import XGBClassifier
 
# File handling
import os


# Creating a new respository to store new dfs
PATH_DFs = './new_dfs'
if not os.path.exists(PATH_DFs):
    os.makedirs(PATH_DFs)

# Creating a new respository to save models results
PATH_train_models = './models'
if not os.path.exists(PATH_train_models):
    os.makedirs(PATH_train_models)

# Creating a new respository to save study plots 
PATH_Plots = './plots_saved'
if not os.path.exists(PATH_Plots):
    os.makedirs(PATH_Plots)

# Setting seed for reproductibility
seed = 42
    
def pre_processing_train_data(train_set, test_set, df_PATH, plot_PATH) :
    
    """
    This function performs several pre-processing steps on the train and test datasets. 
    It computes and saves skewness before and after transformations, applies Yeo-Johnson 
    transformation, scales continuous features using MinMaxScaler, and optionally handles 
    categorical features by transforming them into continuous variables. It also generates 
    and saves histograms to visualize the distributions before and after scaling.

    Parameters:
    - categorical_to_continus: Boolean to enable transformation of categorical features into continuous variables.
    - categorical_to_continus_no_pb: Boolean to replace categorical feature values with percentages (no probability).
    - categorical_to_continus_with_pb: Boolean to replace categorical feature values with probabilities (with a chance).
    """

    #Keeping original data as backup
    original_train = train_set
    original_test = test_set
    
    #Using abs values for negative values
    train_set['Vertical_Distance_To_Hydrology'] = abs(train_set['Vertical_Distance_To_Hydrology'])
    test_set['Vertical_Distance_To_Hydrology'] = abs(test_set['Vertical_Distance_To_Hydrology'])

    # Computes the skewness before transformation
    before_train_skewness = train_set.iloc[:, :9].skew()
    before_test_skewness = test_set.iloc[:, :9].skew()

    # Create a DataFrame to store skewness values
    skewness_df = pd.DataFrame({
        'Feature': before_train_skewness.index,
        'Train_Skewness_Before': before_train_skewness.values,
        'Test_Skewness_Before': before_test_skewness.values
    })

    def transform_yeo_johnson(train_data, test_data):
        # Initialize the PowerTransformer
        pt = PowerTransformer(method='yeo-johnson')
        
        # Fit the transformer on the training data and transform it
        train_data.iloc[:, :9] = pt.fit_transform(train_data.iloc[:, :9])
        
        # Transform the test data using the same transformer (do not fit on test data)
        test_data.iloc[:, :9] = pt.transform(test_data.iloc[:, :9])

    # Apply Yeo-Johnson transformation
    transform_yeo_johnson(train_set, test_set)

    # Check skewness after transformation
    after_train_skewness = train_set.iloc[:, :9].skew()
    after_test_skewness = test_set.iloc[:, :9].skew()

    # Add after transformation skewness to the DataFrame
    skewness_df['Train_Skewness_After'] = after_train_skewness.values
    skewness_df['Test_Skewness_After'] = after_test_skewness.values

    # Save the skewness DataFrame to a CSV file
    skewness_df.to_csv(f"{df_PATH}/skewness_before_after_transformation.csv", index=False)

    # Function to plot and save histograms
    def plot_histograms(data, title, filename):
        plt.figure(figsize=(18, 10))
        data.hist(bins=50, edgecolor='lightblue')
        plt.suptitle(title, fontsize=10)
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
        plt.savefig(filename)
        plt.close()

    # Save histograms before scaling for train dataset
    plot_histograms(original_train.iloc[:, :9], 'Distribution of Continuous Features - Train Dataset (Before Scaling)', f"{plot_PATH}/train_continuous_features_before_scaling.jpeg")

    # Save histograms before scaling for test dataset
    plot_histograms(original_test.iloc[:, :9], 'Distribution of Continuous Features - Test Dataset (Before Scaling)', f"{plot_PATH}/test_continuous_features_before_scaling.jpeg")

    # Apply MinMax scaling
    def scale(train_data, test_data):
        # Initialize the scaler
        scaler = MinMaxScaler()
        # Ensure the first 9 columns are converted to float64 before scaling
        train_data.iloc[:, :9] = train_data.iloc[:, :9].astype('float64')
        test_data.iloc[:, :9] = test_data.iloc[:, :9].astype('float64')
        # Fit the scaler on the training data and transform it
        train_data.iloc[:, :9] = scaler.fit_transform(train_data.iloc[:, :9])
        
        # Transform the test data using the same scaler (do not fit on test data)
        test_data.iloc[:, :9] = scaler.transform(test_data.iloc[:, :9])
        
    # Apply the transformation to both train and test datasets
    scale(train_set, test_set)

    # Save histograms after scaling for train dataset
    plot_histograms(train_set.iloc[:, :9], 'Distribution of Continuous Features - Train Dataset (After Scaling)', f"{plot_PATH}/train_continuous_features_after_scaling.jpeg")

    # Save histograms after scaling for test dataset
    plot_histograms(test_set.iloc[:, :9], 'Distribution of Continuous Features - Test Dataset (After Scaling)', f"{plot_PATH}/test_continuous_features_after_scaling.jpeg")
    
    return train_set, test_set
            
def get_splitted_train_test_data(train_set, data_basic=False, data_for_XBG_Boost=False):
    
    X = train_set.drop(columns=['Cover_Type'])  # Drop the target variable
    y = train_set['Cover_Type']  # Target variable
    
    # Initialize the RandomOverSampler with the specified sampling strategy
    ros = RandomOverSampler(sampling_strategy={1: 36410, 2: 48676, 3: 10000, 4: 1000, 5: 2500, 6: 5000, 7: 6000})
        
    # Apply upsampling to the features and target
    X_resampled, y_resampled = ros.fit_resample(X, y)
        
    # Create a new balanced DataFrame
    train_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    train_balanced['Cover_Type'] = y_resampled

    # Update train 
    train = train_balanced

    if data_basic == True:
        
        """
        This section handles the basic data split by defining the features (X) and the target (y).
        Features (X) are obtained by dropping the 'Cover_Type' column, and the target variable (y) is 'Cover_Type'.
        The data is split into training and testing sets with an 80-20 split, where the training data is used to 
        train the model, and the testing data is used to evaluate it.
        """
       
        X = train.drop(columns=['Cover_Type'])  # Drop the target variable
        y = train['Cover_Type']  # Target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,stratify=y)
        
        return X_train, X_test, y_train, y_test 
        
    if data_for_XBG_Boost == True:
        """
        This section prepares the data for XGBoost by defining the features (X) and the target (y).
        The target variable (y) is adjusted by subtracting 1 from all class labels, which is required 
        for XGBoost's classification.
        The data is split into training and testing sets, and the target variable is modified for use with XGBoost.
        """
        
        # Define features and target
        X = train_set.drop(columns=['Cover_Type'])  # Drop the target variable
        y = train_set['Cover_Type']  # Target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,stratify=y)

        # Adjust the target variable for XGBClassifier (subtract 1 from all class labels)
        y_train_xgb = y_train - 1  # Subtract 1 from all class labels
        y_test_xgb = y_test - 1  # Same adjustment for the test set
        
        return X_train, X_test, y_train_xgb, y_test_xgb    
    
