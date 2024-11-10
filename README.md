# Supervised Learning and Data Challenge  
**November 2024**

## Project Overview  
This project applies supervised learning techniques to tackle two distinct machine learning challenges: a **regression** problem and a **multiclass classification** problem. The work is part of the **Advanced Supervised Learning and Data Challenge** within the **second year** of the Master of Mathematics and AI program at **Paris-Saclay University**.

### 1. **Regression Problem**  
Our objective was to predict the value of **taxi tips** in New York City, a regression task where the target variable is continuous.

### 2. **Multiclass Classification Problem**  
The aim was to predict different **cover types**, a classification task where the target variable has multiple categories.

We tackled each problem independently as follows:

- **Data Analysis**: We began by analyzing the dataset to identify significant variables and made necessary modifications to obtain an optimal dataset for model training.
- **Model Training**: We trained three models on the prepared data.
- **Results and Perspectives**: Finally, we discussed the results and provided perspectives for potential improvements.

## Methodology  
The methodology for both tasks involves:

- **Data Preprocessing**: Handling missing values, encoding categorical variables, feature scaling, and splitting datasets into training and testing sets.
- **Model Training**: Using a variety of supervised learning algorithms.
- **Evaluation**: Assessing model performance using appropriate metrics such as **R squared** for regression, and **F1-score** for classification tasks.
- **Optimization**: Tuning hyperparameters using techniques such as **Grid Search** and **Cross-validation**.

Detailed explanations of the methodology, code implementation, and results can be found in the project **report** located within this repository.

## Running the Code  
To be able to run the code, you need to modify a few lines of the path to the datasets on your local disk in the files in each challenge folder. Also we assume that you already have access to the datasets. 

The lines to be changed are indicated in the README.md file in each folder. Once the paths are updated, you can run the following scripts available for each challenge:

- **explore_data.py**: This script is for the data analysis part.
- **main_tips.py or main_trees.py**: These scripts contain the model training and testing.
- **pre_process_data.py**: This script contains the preprocessing steps.

## Required Python Libraries  
Here’s the list of Python libraries required to run the code:

- `pandas`  
- `numpy`  
- `seaborn`  
- `matplotlib`  
- `xgboost`  
- `scikit-learn`  
- `keras`  
- `os`  

Ensure that all the necessary libraries are installed in your Python environment before running the code. You can install them using `pip`:

```bash
pip install pandas numpy seaborn matplotlib xgboost scikit-learn keras
```

## Authors  
- **Huiqi Vicky ZHENG**
- **Arnaud FEYEL**
