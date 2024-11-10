## Running the Code

Before running the code, you need to modify the paths to the datasets in the following scripts:

In `explore_data.py`, locate the following lines (lines 18 and 19 ):

```python
# Import train and test data
test = pd.read_csv('/path/to/your/test.csv', index_col=0)
train = pd.read_csv('/path/to/your/train.csv', index_col=0)
```

In `main_trees.py`, locate the following lines (lines 45 and 46 ): 

```python
# Import train and test data
test = pd.read_csv('/path/to/your/test.csv', index_col=0)
train = pd.read_csv('/path/to/your/train.csv', index_col=0)
```
