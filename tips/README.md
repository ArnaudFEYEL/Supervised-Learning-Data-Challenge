## Running the Code

Before running the code, you need to modify the paths to the datasets in the following scripts:

In `explore_data.py`, locate the following lines (lines 23 and 24 ):

```python
# Import train and test data
test = pd.read_parquet('/path/to/your/test.parquet')
train = pd.read_parquet('/path/to/your/train.parquet')
```

In `main_tips.py`, locate the following lines (lines 57 and 58 ): 

```python
# Import train and test data
test = pd.read_parquet('/path/to/your/test.parquet')
train = pd.read_parquet('/path/to/your/train.parquet')
```
