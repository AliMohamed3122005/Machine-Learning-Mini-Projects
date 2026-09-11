import pandas as pd
import kagglehub
from pathlib import Path
from model import train_model

# Download dataset
path = kagglehub.dataset_download("uciml/mushroom-classification")

# Convert path to Path object
data_path = Path(path)

print("Dataset path:", data_path)

# Read CSV file
df = pd.read_csv(data_path / "mushrooms.csv")


print(df.shape)
print('------------------------------')
print(df.iloc[0:5,:])
print('------------------------------')
print(df.info())
print('------------------------------') 
print(df.describe())
print('------------------------------')
print(df.isnull().sum())
print('------------------------------')


df = df.astype('category')
print(df.info())

train_model(df)