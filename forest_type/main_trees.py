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

# Import custom code for data pre processing
from pre_process_data import pre_processing_train_data, get_splitted_train_test_data
from explore_data import drop_missing_values

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
    
# Import train and test data
test = pd.read_csv('test.csv', index_col=0)
train = pd.read_csv('train.csv', index_col=0)
print("Data Imported !")

# Setting seed for reproductibility
seed = 42

# Pre processing data   
drop_missing_values(train, drop=True)
train, test = pre_processing_train_data(train, test, PATH_DFs, PATH_Plots)


def train_model(train_data, svm_one_vs_rest=False, basic_RF=False, xgb_RF=False, stacking=False, voting=False):
    
    if svm_one_vs_rest == True:
        """
        This section trains a OneVsRest SVM model using a SGDClassifier (hinge loss).
        It splits the data into training and testing sets and then fits the model to the training data.
        Afterward, it makes predictions and evaluates the model using classification report, confusion matrix,
        mean squared error, R² scores, and cross-validation.
        The results are saved to a text file.
        """
        X_train, X_test, y_train, y_test = get_splitted_train_test_data(train_data, data_basic= True, data_for_XBG_Boost=False)

        # Logistic Regression Classifier
        svm_model = OneVsRestClassifier(SGDClassifier(loss='hinge', random_state=seed))
        print(f"training SVM model")
        
        svm_model.fit(X_train, y_train)  # Fit the model

        # Make predictions
        lr_predictions = svm_model.predict(X_test)

        # Evaluate the model
        lr_classification_rep = classification_report(y_test, lr_predictions)
        lr_confusion_mat = confusion_matrix(y_test, lr_predictions)

        # Computing metrics
        lr_mse = mean_squared_error(y_test, lr_predictions)
        lr_r2_train = svm_model.score(X_train, y_train)
        lr_r2_test = r2_score(y_test, lr_predictions)
        lr_cross_val = cross_val_score(svm_model, X_train, y_train, cv=5).mean()

        # Save the results to a text file
        lr_filename = "model_evaluation_SVM_One_Vs_Rest.txt"
        with open(f"{PATH_train_models}/{lr_filename}", "w") as file:
            file.write(f"Logistic Regression Classification Report:\n{lr_classification_rep}\n")
            file.write(f"Confusion Matrix:\n{lr_confusion_mat}\n\n")
            file.write(f"LogisticRegression:\n")
            file.write(f'Mean Squared Error: {lr_mse:.2f}\n')
            file.write(f'R² on Training Set: {lr_r2_train:.2f}\n')
            file.write(f'R² on Test Set: {lr_r2_test:.2f}\n')
            file.write(f'Cross-validation score: {lr_cross_val}\n\n')

        print(f"Results saved to {lr_filename}")
        
        return svm_model
    
    if basic_RF == True :
        """
        This section trains a Random Forest Classifier with a specified number of estimators and max depth.
        It splits the data into training and testing sets and fits the model to the training data.
        Afterward, it makes predictions and evaluates the model using classification report, confusion matrix,
        mean squared error, R² scores, and cross-validation.
        Additionally, it computes feature importance and identifies the most significant features.
        The results are saved to a text file.
        """
        X_train, X_test, y_train, y_test = get_splitted_train_test_data(train_data, data_basic= True, data_for_XBG_Boost=False)

        # Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=seed)
        print(f"training RF model")
        rf_model.fit(X_train, y_train)  # Fit the model

        # Make predictions
        rf_predictions = rf_model.predict(X_test)

        # Evaluate the model
        rf_classification_rep = classification_report(y_test, rf_predictions)
        rf_confusion_mat = confusion_matrix(y_test, rf_predictions)

        # Computing metrics
        rf_mse = mean_squared_error(y_test, rf_predictions)
        rf_r2_train = rf_model.score(X_train, y_train)
        rf_r2_test = r2_score(y_test, rf_predictions)
        rf_cross_val = cross_val_score(rf_model, X_train, y_train, cv=5).mean()

        # Identify the most significant features
        rf_feature_importance = rf_model.feature_importances_
        features = X_train.columns
        rf_top_features = sorted(zip(features, rf_feature_importance), key=lambda x: x[1], reverse=True)[:5]


        # Save the results to a text file
        rf_filename = "model_evaluation_RandomForestClassifier.txt"
        with open(f"{PATH_train_models}/{rf_filename}", "w") as file:
            file.write(f"Random Forest Classification Report:\n{rf_classification_rep}\n")
            file.write(f"Confusion Matrix:\n{rf_confusion_mat}\n\n")
            file.write(f"RandomForestClassifier:\n")
            file.write(f'Mean Squared Error: {rf_mse:.2f}\n')
            file.write(f'R² on Training Set: {rf_r2_train:.2f}\n')
            file.write(f'R² on Test Set: {rf_r2_test:.2f}\n')
            file.write(f'Cross-validation score: {rf_cross_val}\n\n')
            for feature, importance in rf_top_features:
                file.write(f"{feature}: {importance:.4f}\n")
        print(f"Results saved to {rf_filename}")
        
        return rf_model
        
    if xgb_RF == True:
        """
        This section trains an XGBoost Classifier (XGBClassifier) model.
        It first splits the data for XGBoost using the specified function and parameters.
        Then, it initializes the XGBClassifier with specified hyperparameters, fits the model to the training data,
        makes predictions, and evaluates the model using classification report, confusion matrix,
        mean squared error, R² scores, and cross-validation.
        Additionally, it computes feature importance and identifies the most significant features.
        The results are saved to a text file.
        """
        # Split data for XGBoost
        X_train, X_test, y_train, y_test = get_splitted_train_test_data(train_data, data_basic=False, data_for_XBG_Boost=True)

        # XGB Classifier
        xgb_model = XGBClassifier(n_estimators=200, max_depth=20, eval_metric='mlogloss', random_state=seed)
        print(f"training XGB model")
        xgb_model.fit(X_train, y_train)  # Fit the model

        # Make predictions
        xgb_predictions = xgb_model.predict(X_test)

        # Evaluate the model
        xgb_classification_rep = classification_report(y_test, xgb_predictions)
        xgb_confusion_mat = confusion_matrix(y_test, xgb_predictions)

        # Computing metrics
        xgb_mse = mean_squared_error(y_test, xgb_predictions)
        xgb_r2_train = xgb_model.score(X_train, y_train)
        xgb_r2_test = r2_score(y_test, xgb_predictions)
        xgb_cross_val = cross_val_score(xgb_model, X_train, y_train, cv=5).mean()

        # Identify the most significant features
        xgb_feature_importance = xgb_model.feature_importances_
        features = X_train.columns
        xgb_top_features = sorted(zip(features, xgb_feature_importance), key=lambda x: x[1], reverse=True)[:5]

        # Save the results to a text file
        xgb_filename = "model_evaluation_XGBClassifier.txt"
        with open(f"{PATH_train_models}/{xgb_filename}", "w") as file:
            file.write(f"XGB Classification Report:\n{xgb_classification_rep}\n")
            file.write(f"Confusion Matrix:\n{xgb_confusion_mat}\n\n")
            file.write(f"XGBClassifier:\n")
            file.write(f'Mean Squared Error: {xgb_mse:.2f}\n')
            file.write(f'R² on Training Set: {xgb_r2_train:.2f}\n')
            file.write(f'R² on Test Set: {xgb_r2_test:.2f}\n')
            file.write(f'Cross-validation score: {xgb_cross_val}\n\n')
            file.write(f"XGBClassifier - Top 5 Feature Importance:\n")
            for feature, importance in xgb_top_features:
                file.write(f"{feature}: {importance:.4f}\n")
        
        print(f"Results saved to {xgb_filename}")
        
        return xgb_model

    # Stacking Model Section
    if stacking == True:
        """
        This section trains a StackingClassifier with multiple base classifiers (SVM, RF, XGBoost)
        and a meta-model (RF).
        """

        # Train base models
        X_train, X_test, y_train, y_test = get_splitted_train_test_data(train_data, data_basic=True, data_for_XBG_Boost=False)

        y_train = y_train - 1  # Adjust for XGBoost

        base_learners = [
            ('svm', OneVsRestClassifier(SGDClassifier(loss='hinge', random_state=seed))),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=20, random_state=seed)),
            ('xgb', XGBClassifier(n_estimators=200, max_depth=20, eval_metric='mlogloss', random_state=seed))
        ]

        # Meta-learner (RF)
        meta_learner = RandomForestClassifier(n_estimators=100)

        # Create and train the Stacking Classifier
        stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)
        print(f"training stacking model")
        stacking_model.fit(X_train, y_train)

        # Make predictions
        stacking_predictions = stacking_model.predict(X_test) + 1  # Adding 1 to match original target range

        # Evaluate the model
        stacking_classification_rep = classification_report(y_test, stacking_predictions)
        stacking_confusion_mat = confusion_matrix(y_test, stacking_predictions)

        # Computing metrics
        stacking_mse = mean_squared_error(y_test, stacking_predictions)
        stacking_r2_train = stacking_model.score(X_train, y_train)
        stacking_r2_test = r2_score(y_test, stacking_predictions)
        stacking_cross_val = cross_val_score(stacking_model, X_train, y_train, cv=5).mean()

        # Save the results to a text file
        stacking_filename = "model_evaluation_StackingClassifier.txt"
        with open(f"{PATH_train_models}/{stacking_filename}", "w") as file:
            file.write(f"Stacking Classifier Classification Report:\n{stacking_classification_rep}\n")
            file.write(f"Confusion Matrix:\n{stacking_confusion_mat}\n\n")
            file.write(f"StackingClassifier:\n")
            file.write(f'Mean Squared Error: {stacking_mse:.2f}\n')
            file.write(f'R² on Training Set: {stacking_r2_train:.2f}\n')
            file.write(f'R² on Test Set: {stacking_r2_test:.2f}\n')
            file.write(f'Cross-validation score: {stacking_cross_val}\n\n')

        print(f"Results saved to {stacking_filename}")

        return stacking_model

    if voting == True :
        """
        This section trains a VotingClassifier using a combination of base classifiers (SVM, RF, XGBoost) 
        with soft voting to aggregate predictions.
        """
        # Split data 
        X_train, X_test, y_train, y_test = get_splitted_train_test_data(train_data, data_basic=True, data_for_XBG_Boost=False)
        y_train = y_train - 1  # Adjust for XGBoost
        # Base learners for voting classifier
        base_learners = [
            ('svm', OneVsRestClassifier(SVC(kernel='linear', random_state=seed))),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=20, random_state=seed)),
            ('xgb', XGBClassifier(n_estimators=200, max_depth=20, eval_metric='mlogloss', random_state=seed))
        ]
        # Voting Classifier with Soft Voting
        voting_model = VotingClassifier(estimators=base_learners, voting='soft')

        # Train the Voting Classifier
        print(f"training voting model")
        voting_model.fit(X_train, y_train)

        # Make predictions
        voting_predictions = voting_model.predict(X_test) + 1  # Adding 1 to match original target range

        # Evaluate the model
        voting_classification_rep = classification_report(y_test, voting_predictions)
        voting_confusion_mat = confusion_matrix(y_test, voting_predictions)

        # Computing metrics
        voting_mse = mean_squared_error(y_test, voting_predictions)
        voting_r2_train = voting_model.score(X_train, y_train)
        voting_r2_test = r2_score(y_test, voting_predictions)
        voting_cross_val = cross_val_score(voting_model, X_train, y_train, cv=5).mean()

        # Save the results to a text file
        voting_filename = "model_evaluation_VotingClassifier.txt"
        with open(f"{PATH_train_models}/{voting_filename}", "w") as file:
            file.write(f"Voting Classifier Classification Report:\n{voting_classification_rep}\n")
            file.write(f"Confusion Matrix:\n{voting_confusion_mat}\n\n")
            file.write(f"VotingClassifier:\n")
            file.write(f'Mean Squared Error: {voting_mse:.2f}\n')
            file.write(f'R² on Training Set: {voting_r2_train:.2f}\n')
            file.write(f'R² on Test Set: {voting_r2_test:.2f}\n')
            file.write(f'Cross-validation score: {voting_cross_val}\n\n')

        print(f"Results saved to {voting_filename}")

        return voting_model

def make_test_prediction(svm_one_vs_rest=False, basic_RF=False, xgb_RF=False, stacking=False, voting=False):
    """
    This function is used to make predictions on the test dataset using different trained models (SVM One-vs-Rest, Random Forest, XGBoost, Stacking or Voting Model).
    Based on the input flags, it selects the corresponding model, makes predictions, adjusts the predicted values if necessary, 
    and then saves the predictions to a CSV file for submission.
    
    Parameters:
    - svm_one_vs_rest (bool): If True, uses the SVM One-vs-Rest model for prediction.
    - basic_RF (bool): If True, uses the Random Forest model for prediction.
    - xgb_RF (bool): If True, uses the XGBoost model for prediction.
    - stacking (bool): If True, uses a stacking ensemble model for prediction.
    - voting (bool): If True, uses a voting ensemble model for prediction.
    """
    
    if svm_one_vs_rest == True :
        
        """
        Trains and uses the SVM One-vs-Rest model to make predictions on the test dataset. 
        Saves the predictions to 'svm_one_vs_rest_model_submission.csv'.
        """
        
        trained_model = train_model(train, 
                                    svm_one_vs_rest=True, 
                                    basic_RF=False, 
                                    xgb_RF=False,
                                    stacking=False, 
                                    voting=False)

        # Make predictions
        model_pred = trained_model.predict(test)

        # Predict on the test dataset
        predictions_balanced_transformed = model_pred  # Use the adjusted predictions
        
        # Create a submission DataFrame
        submission = pd.DataFrame({
            'row_ID': range(len(predictions_balanced_transformed)),  # Adjust if row_ID is different
            'Cover_Type': predictions_balanced_transformed
        })
        # Save the submission DataFrame to CSV
        submission.to_csv('svm_one_vs_rest_model_submission.csv', index=False)
        
    if basic_RF == True:
        
        """
        Trains and uses the Random Forest model to make predictions on the test dataset.
        Saves the predictions to 'RF_model_submission.csv'.
        """
        
        trained_model = train_model(train, 
                                    svm_one_vs_rest=False, 
                                    basic_RF=True, 
                                    xgb_RF=False,
                                    stacking=False, 
                                    voting=False)

        # Make predictions
        model_pred = trained_model.predict(test)

        # Predict on the test dataset
        predictions_balanced_transformed = model_pred  # Use the adjusted predictions
        
        # Create a submission DataFrame
        submission = pd.DataFrame({
            'row_ID': range(len(predictions_balanced_transformed)),  # Adjust if row_ID is different
            'Cover_Type': predictions_balanced_transformed
        })
        
        # Save the submission DataFrame to CSV
        submission.to_csv('RF_model_submission.csv', index=False)
        
    if xgb_RF == True:
        
        """
        Trains and uses the XGBoost model to make predictions on the test dataset.
        Adjusts the predictions to match the original target range (1 to 7).
        Saves the predictions to 'XGB_model_submission.csv'.
        """
        
        trained_model = train_model(train, 
                                    svm_one_vs_rest=False, 
                                    basic_RF=False, 
                                    xgb_RF=True, 
                                    stacking=False, 
                                    voting=False)

        # Make predictions
        model_pred = trained_model.predict(test)

        # Adjust predictions since they start from 0, but your classes are from 1 to 7
        model_pred_adjusted = model_pred + 1  # Add 1 to match your target range (1 to 7)

        # Predict on the test dataset
        predictions_balanced_transformed = model_pred_adjusted  # Use the adjusted predictions
        
        # Create a submission DataFrame
        submission = pd.DataFrame({
            'row_ID': range(len(predictions_balanced_transformed)),  # Adjust if row_ID is different
            'Cover_Type': predictions_balanced_transformed
        })
        
        # Save the submission DataFrame to CSV
        submission.to_csv('XGB_model_submission.csv', index=False)

    if stacking == True:
        # Train the stacking model with specified flags
        trained_model = train_model(train, 
                                    svm_one_vs_rest=False, 
                                    basic_RF=False, 
                                    xgb_RF=False, 
                                    stacking=True, 
                                    voting=False
                                )
        
        # Make predictions on the test dataset using the trained stacking model
        model_pred = trained_model.predict(test) + 1  # Adjust predictions as needed by adding 1
        
        # Store the adjusted predictions for final submission
        predictions_balanced_transformed = model_pred
        
        # Create a DataFrame to hold the predictions with row IDs
        submission = pd.DataFrame({
            'row_ID': range(len(predictions_balanced_transformed)), 
            'Cover_Type': predictions_balanced_transformed
        })
        
        # Save the predictions to a CSV file named 'stacking_model_submission.csv' for submission
        submission.to_csv('stacking_model_submission.csv', index=False)
    
    if voting == True:
        
        # Train the voting ensemble model with specified flags
        trained_model = train_model(train, 
                                    svm_one_vs_rest=False, 
                                    basic_RF=False, 
                                    xgb_RF=False, 
                                    stacking=False, 
                                    voting=True
                                )
        
        # Make predictions on the test dataset using the trained voting model
        model_pred = trained_model.predict(test) + 1  # Adjust predictions as needed by adding 1
        
        # Store the adjusted predictions for final submission
        predictions_balanced_transformed = model_pred
        
        # Create a DataFrame to hold the predictions with row IDs
        submission = pd.DataFrame({
            'row_ID': range(len(predictions_balanced_transformed)), 
            'Cover_Type': predictions_balanced_transformed
        })
        
        # Save the predictions to a CSV file named 'voting_model_submission.csv' for submission
        submission.to_csv('voting_model_submission.csv', index=False)

# Main execution block
if __name__ == "__main__":
    make_test_prediction(svm_one_vs_rest=False, 
                         basic_RF=False, 
                         xgb_RF=False, 
                         stacking=False, 
                         voting=True)
    
