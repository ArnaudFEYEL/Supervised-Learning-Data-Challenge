import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Data manipulation
import numpy as np
import pandas as pd
 
# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn models and utilities
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import BayesianRidge
from keras import models, layers, optimizers, callbacks

# XGBoost
from xgboost import XGBRegressor

# lightgbm
from lightgbm import LGBMRegressor

# Import custom local code
from pre_process_data import pre_process_data, drop_outliers

# File handling
import os



# Creating a new respository to save study plots 
PATH_Plots = './plots_saved'
if not os.path.exists(PATH_Plots):
    os.makedirs(PATH_Plots)

# Creating a new respository to store new dfs
PATH_DFs = './new_dfs'
if not os.path.exists(PATH_DFs):
    os.makedirs(PATH_DFs)

# Creating a new respository to save models results
PATH_train_models = './models'
if not os.path.exists(PATH_train_models):
    os.makedirs(PATH_train_models)
    

# Import train and test data
test = pd.read_parquet('/path/to/your/test.parquet')
train = pd.read_parquet('/path/to/your/train.parquet')

print("Data Imported !")

# Setting seed for reproductibility
seed = 42

train, test = pre_process_data(train, test, PATH_DFs, transform_scale=True)
train_data = drop_outliers(train)

def train_model(bayesian_R=False, basic_RF=False, xgb_RF=False, stacking_model=False, neuron_network=False):

    # Define features and target
    y = train_data['tip_amount']  # Target variable
    X = train_data.drop(columns=['tip_amount'])  # Drop the target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    if bayesian_R == True:
        """
        This section trains a Bayesian Ridge Regressor with kernel approximation.
        It splits the data into training and testing sets and fits the model to the training data.
        Afterward, it makes predictions and evaluates the model using mean squared error,
        R² scores, and cross-validation.
        The results are saved to a text file.
        """

        model = BayesianRidge(
                        max_iter=10000,       # Number of iterations, adjust for complexity
                        tol=1e-4,             # Tolerance for convergence
                        alpha_1=1e-6,         # Hyperparameters for prior over the alpha parameter
                        alpha_2=1e-8,         # Can be tuned based on data complexity
                        lambda_1=1e-6,        # Hyperparameters for prior over the lambda parameter
                        lambda_2=1e-8         # Similar to alpha parameters
                    )
        
        # Fit the model
        print(f"training bayesian ridge model")
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        r2_train = r2_score(y_train, model.predict(X_train))  # R² on the training set
        r2_test = r2_score(y_test, predictions)               # R² on the test set

        # Save the results to a text file
        results_filename = "model_evaluation_Bayesian_Ridge.txt"
        with open(f"{PATH_train_models}/{results_filename}", "w") as file:
            file.write(f"Bayesian Ridge Regression Model Evaluation:\n")
            file.write(f'Mean Squared Error: {mse:.2f}\n')
            file.write(f'R² on Training Set: {r2_train:.2f}\n')
            file.write(f'R² on Test Set: {r2_test:.2f}\n')

        print(f"Results saved to {results_filename}")

        return model
        
    
    if basic_RF == True :
        """
        This section trains a Random Forest Classifier with a specified number of estimators and max depth.
        It splits the data into training and testing sets and fits the model to the training data.
        Afterward, it makes predictions and evaluates the model using classification report, confusion matrix,
        mean squared error, R² scores, and cross-validation.
        Additionally, it computes feature importance and identifies the most significant features.
        The results are saved to a text file.
        """

        # Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=seed)
        rf_model.fit(X_train, y_train)  # Fit the model

        # Make predictions
        print(f"training RF model")
        rf_predictions = rf_model.predict(X_test)
  
        # Evaluate the model
        rf_mse = mean_squared_error(y_test, rf_predictions)
        rf_r2_train = rf_model.score(X_train, y_train)  # R² on the training set
        rf_r2_test = r2_score(y_test, rf_predictions)   # R² on the test set
        rf_cross_val = cross_val_score(rf_model, X_train, y_train, cv=5).mean()

        # Identify the most significant features
        rf_feature_importance = rf_model.feature_importances_
        features = X_train.columns
        rf_top_features = sorted(zip(features, rf_feature_importance), key=lambda x: x[1], reverse=True)[:5]

        # Save the results to a text file
        rf_filename = "model_evaluation_RandomForestRegressor.txt"
        with open(f"{PATH_train_models}/{rf_filename}", "w") as file:
            file.write(f"Random Forest Model Evaluation:\n")
            file.write(f'Mean Squared Error: {rf_mse:.2f}\n')
            file.write(f'R² on Training Set: {rf_r2_train:.2f}\n')
            file.write(f'R² on Test Set: {rf_r2_test:.2f}\n')
            file.write(f'Cross-validation score: {rf_cross_val:.4f}\n\n')

            # Write feature importance to the file
            for feature, importance in rf_top_features:
                file.write(f"{feature}: {importance:.4f}\n")

        print(f"Results saved to {rf_filename}")

        return rf_model
        
    if xgb_RF == True:
        """
        This section trains an XGBoost Classifier (XGBClassifier) model.
        It first splits the data for XGBoost using the specified function and parameters.
        Then, it initializes the XGBClassifier with specified hyperparameters, fits the model to the training data,
        makes predictions, and evaluates the model using classification report,
        mean squared error, R² scores, and cross-validation.
        The results are saved to a text file.
        """
        
        # XGB Classifier
        xgb_model = XGBRegressor(n_estimators=50, 
                                 max_depth=4, 
                                 eval_metric='rmse', 
                                 random_state=seed)
        
        print(f"training XGB model")
        xgb_model.fit(X_train, y_train)  # Fit the model
        
        xgb_predictions = xgb_model.predict(X_test)

        # Computing regression metrics
        xgb_mse = mean_squared_error(y_test, xgb_predictions)
        xgb_r2_train = xgb_model.score(X_train, y_train)  # R² on training set
        xgb_r2_test = r2_score(y_test, xgb_predictions)   # R² on test set
        xgb_cross_val = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2').mean()  # R² cross-validation score

        # Identify the most significant features
        xgb_feature_importance = xgb_model.feature_importances_
        features = X_train.columns
        xgb_top_features = sorted(zip(features, xgb_feature_importance), key=lambda x: x[1], reverse=True)[:5]

        # Save the results to a text file
        xgb_filename = "model_evaluation_XGBRegressor.txt"
        with open(f"{PATH_train_models}/{xgb_filename}", "w") as file:
            file.write(f"XGB Regressor Evaluation:\n")
            file.write(f'Mean Squared Error: {xgb_mse:.2f}\n')
            file.write(f'R² on Training Set: {xgb_r2_train:.2f}\n')
            file.write(f'R² on Test Set: {xgb_r2_test:.2f}\n')
            file.write(f'Cross-validation R² score: {xgb_cross_val:.2f}\n\n')
            file.write(f"XGB Regressor - Top 5 Feature Importance:\n")
            for feature, importance in xgb_top_features:
                file.write(f"{feature}: {importance:.4f}\n")

        print(f"Results saved to {xgb_filename}")
        
        return xgb_model
    
    if stacking_model == True:
        """
        This section trains a Stacking Regressor with a combination of base models (Bayesian Ridge, XGBoost, Random Forest)
        and a meta-model (Random Forest) for final predictions. It splits the data into training and testing sets,
        fits the model to the training data, makes predictions, and evaluates the model using mean squared error,
        R² scores, and cross-validation. The results are saved to a text file.
        """
        # Initialize base models
        model_bayesian_ridge = BayesianRidge(max_iter=10000, 
                                             tol=1e-4, 
                                             alpha_1=1e-6,
                                             alpha_2=1e-8, 
                                             lambda_1=1e-6, 
                                             lambda_2=1e-8
        )
        
        model_rf = RandomForestRegressor(n_estimators=150, 
                                         max_depth=20, 
                                         random_state=seed)
     
        model_xgb = XGBRegressor(n_estimators=50, 
                                 max_depth=4, 
                                 eval_metric='rmse', 
                                 random_state=seed)
     
        model_lgbm = LGBMRegressor(n_estimators=50, 
                                   max_depth=4, 
                                   random_state=seed)
     
        model_elastic_net = ElasticNet(alpha=0.1, 
                                       l1_ratio=0.5, 
                                       random_state=seed)

        # Stacking Regressor with Random Forest as the meta-estimator
        stacking_reg = StackingRegressor(
            estimators=[('bayesian_ridge', model_bayesian_ridge), 
                        ('xgb', model_xgb), 
                        ('random_forest', model_rf),
                        ('lightgbm', model_lgbm),
                        ('elastic_net', model_elastic_net)],
            final_estimator=RandomForestRegressor(n_estimators=50, max_depth=5, random_state=seed)
        )
        print(f"Training stacking model")
        stacking_reg.fit(X_train, y_train)
        
        # Predictions and Evaluation
        predictions = stacking_reg.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2_train = stacking_reg.score(X_train, y_train)
        r2_test = r2_score(y_test, predictions)
        cross_val_score_stacking = cross_val_score(stacking_reg, X_train, y_train, cv=5, scoring='r2').mean()

        # Save results to a text file
        stacking_filename = "model_evaluation_Stacking.txt"
        with open(f"{PATH_train_models}/{stacking_filename}", "w") as file:
            file.write(f"Stacking Regressor Evaluation:\n")
            file.write(f'Mean Squared Error: {mse:.2f}\n')
            file.write(f'R² on Training Set: {r2_train:.2f}\n')
            file.write(f'R² on Test Set: {r2_test:.2f}\n')
            file.write(f'Cross-validation R² score: {cross_val_score_stacking:.2f}\n')

        print(f"Results saved to {stacking_filename}")

        return stacking_reg
    
    if neuron_network == True :
        """
        This section trains a neural network model with multiple layers using dropout and batch normalization for 
        regularization. It splits the data into training and testing sets, compiles the model with mean squared error 
        as the loss function, and sets up callbacks for early stopping and best model checkpointing.
        The model is trained and evaluated on mean squared error and R² scores. Additionally, the training curve is 
        saved as a plot, and the results are saved to a text file.
        """
        model = models.Sequential([
            layers.Dense(32, activation="tanh", input_shape=(X_train.shape[1],)),  # Input layer with 16 units
            layers.Dropout(0.3),  # Dropout layer to prevent overfitting
            layers.BatchNormalization(),  # Batch normalization to stabilize training
            
            # Hidden layer with 32 units and tanh activation
            layers.Dense(32, activation="tanh"),
            
            layers.Dropout(0.3),  # Dropout layer for regularization
            layers.BatchNormalization(),  # Batch normalization for stability
            
            # Output layer
            layers.Dense(1, activation=None)  # Output layer with a single neuron (no activation)
        ])
        
        # Compile model  
        opt = optimizers.Adam(learning_rate=0.001)    
        model.compile(optimizer=opt, loss="mean_squared_error", metrics=["mean_squared_error"])

        # Callbacks for early stopping and model checkpoint
        callback_list = [
            callbacks.EarlyStopping(monitor="val_loss", patience=100), #Modify the value of patience if needed
            callbacks.ModelCheckpoint(
                filepath="models/aps_model.weights.h5",  # Update file extension
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True
            )
            ]

        print(f"Neuron Network is training")
        # Train the model
        history = model.fit(X_train, y_train, batch_size=16, epochs=100, validation_split=0.2, callbacks=callback_list)

        # Load best weights and make predictions
        model.load_weights('/aps_model.weights.h5')

        # Plot training curves
        val_loss, train_loss = history.history['val_loss'], history.history['loss']
        epochs = list(range(1, len(val_loss) + 1))

        # Apply absolute value
        train_loss = np.abs(train_loss)
        val_loss = np.abs(val_loss)

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
        plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')

        # Set log scale for the y-axis
        plt.yscale('log')

        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Curves (Log Scale)')
        plt.legend()

        # Save the plot
        plt.savefig(f"{PATH_Plots}/training_evolution_Nadam_new_log_scale.jpeg")
        plt.close()
        
        # Predictions and Evaluation
        predictions = model.predict(X_test).flatten()  # Flatten to ensure shape consistency
        mse = mean_squared_error(y_test, predictions)
        r2_test = r2_score(y_test, predictions)
        r2_train = model.evaluate(X_train, y_train, verbose=0)[1]  # Using model's internal evaluation for R² on training set

        # Save results to a text file
        nn_filename = f"{PATH_train_models}/model_evaluation_NeuralNetwork.txt"
        with open(nn_filename, "w") as file:
            file.write("Neural Network Evaluation:\n")
            file.write(f"Mean Squared Error: {mse:.2f}\n")
            file.write(f"R² on Training Set: {r2_train:.2f}\n")
            file.write(f"R² on Test Set: {r2_test:.2f}\n")

        print(f"Results saved to {nn_filename}")

            
        return model 
        

def make_test_prediction(bayesian_R=False, 
                         basic_RF=False, 
                         xgb_RF=False, 
                         stacking_model=False, 
                         neuron_network=False):
    """
    This function is used to make predictions on the test dataset using different trained models (SVM One-vs-Rest, Random Forest, or XGBoost).
    Based on the input flags, it selects the corresponding model, makes predictions, adjusts the predicted values if necessary, 
    and then saves the predictions to a CSV file for submission.
    
    Parameters:
    - svm_one_vs_rest (bool): If True, trains and uses the SVM One-vs-Rest model.
    - basic_RF (bool): If True, trains and uses the Random Forest model.
    - xgb_RF (bool): If True, trains and uses the XGBoost model.
    """
    
    if bayesian_R == True :
        
        """
        Trains and uses the SVM One-vs-Rest model to make predictions on the test dataset. 
        Saves the predictions to 'svm_one_vs_rest_model_submission.csv'.
        """
        
        trained_model = train_model(bayesian_R=True, 
                                    basic_RF=False, 
                                    xgb_RF=False, 
                                    stacking_model=False, 
                                    neuron_network=False)

        # Make predictions
        model_pred = trained_model.predict(test)

        # Predict on the test dataset
        predictions_balanced_transformed = model_pred  # Use the adjusted predictions
        
        # Create a submission DataFrame
        submission = pd.DataFrame({
            'row_ID': range(len(predictions_balanced_transformed)),  # Adjust if row_ID is different
            'tip_amount': predictions_balanced_transformed
        })
        # Save the submission DataFrame to CSV
        submission.to_csv('bayesian_Ridge_model_submission.csv', index=False)
        
    if basic_RF == True:
        
        """
        Trains and uses the Random Forest model to make predictions on the test dataset.
        Saves the predictions to 'RF_model_submission.csv'.
        """
        
        trained_model = train_model(bayesian_R=False, 
                                    basic_RF=True, 
                                    xgb_RF=False, 
                                    stacking_model=False, 
                                    neuron_network=False)
        # Make predictions
        model_pred = trained_model.predict(test)

        # Predict on the test dataset
        predictions_balanced_transformed = model_pred  # Use the adjusted predictions
        
        # Create a submission DataFrame
        submission = pd.DataFrame({
            'row_ID': range(len(predictions_balanced_transformed)),  # Adjust if row_ID is different
            'tip_amount': predictions_balanced_transformed
        })
        
        # Save the submission DataFrame to CSV
        submission.to_csv('RF_model_submission.csv', index=False)
        
    if xgb_RF == True:
        
        """
        Trains and uses the XGBoost model to make predictions on the test dataset.
        Saves the continuous predictions to 'XGB_model_submission.csv'.
        """
        
        # Train the XGBoost model with regression settings
        trained_model = train_model(bayesian_R=False, 
                                    basic_RF=False, 
                                    xgb_RF=True, 
                                    stacking_model=False, 
                                    neuron_network=False)

        # Make predictions on the test dataset
        model_pred = trained_model.predict(test)

        # Create a submission DataFrame
        submission = pd.DataFrame({
            'row_ID': range(len(model_pred)),  # Adjust if row_ID is different
            'tip_amount': model_pred  # Use the continuous predictions directly
        })
        
        # Save the submission DataFrame to CSV
        submission.to_csv('XGB_model_submission.csv', index=False)

        print("Continuous regression predictions saved to XGB_model_submission.csv")

    # Stacking Regressor model
    if stacking_model == True :

        """
        Trains and uses the Stacking Regressor model to make predictions on the test dataset.
        The predictions are saved to a CSV file named 'Stacking_model_submission.csv' for submission.
        """
     
        trained_model = train_model(bayesian_R=False, 
                                    basic_RF=False, 
                                    xgb_RF=False, 
                                    stacking_model=True, 
                                    neuron_network=False)
        
        model_pred = trained_model.predict(test)
        
        # Create and save the submission DataFrame
        submission = pd.DataFrame({'row_ID': range(len(model_pred)), 'tip_amount': model_pred})
        submission.to_csv('Stacking_model_submission.csv', index=False)
        
        print("Predictions saved toStacking_model_submission.csv")

    if neuron_network == True :
     
        """
        Trains and uses the Neural Network model to make predictions on the test dataset.
        The predictions are saved to a CSV file named 'Neuron_network_model_submission.csv' for submission.
        """
     
        trained_model = train_model(bayesian_R=False, 
                                    basic_RF=False, 
                                    xgb_RF=False, 
                                    stacking_model=False, 
                                    neuron_network=True)
        
        model_pred = trained_model.predict(test)
                
        # Ensure model_pred is 1D
        model_pred = model_pred.flatten()  

        # Create the DataFrame
        submission = pd.DataFrame({'row_ID': range(len(model_pred)), 'tip_amount': model_pred})
        submission.to_csv('Neuron_network_model_submission.csv', index=False)
        print("Predictions saved to Neuron_network_model_submission.csv")        

        
    
# Main execution block
if __name__ == "__main__":
    make_test_prediction(bayesian_R=False, 
                         basic_RF=False, 
                         xgb_RF=False, 
                         stacking_model=False, 
                         neuron_network=False)
