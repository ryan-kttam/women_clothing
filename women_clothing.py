import pandas as pd
import numpy as np
data = pd.read_csv('Womens_Clothing_Reviews.csv',header=0)

data = data.rename(columns={list(data)[0] : 'count'})
data.head()

data.describe()