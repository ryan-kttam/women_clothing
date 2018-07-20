import pandas as pd
import numpy as np
data = pd.read_csv('Womens_Clothing_Reviews.csv',header=0)

data = data.rename(columns={list(data)[0] : 'count'})
data.head()

data.describe()

dept_vs_rating = data.groupby(['Department Name', 'Rating']).count()


import seaborn as sns

sns.countplot(x='Department Name', hue='Rating',data=data)

sns.countplot(x='Rating', hue='Recommended IND',data=data)

sns.countplot(x='Department Name', hue='Recommended IND',data=data)

# working
sns.barplot(x='Department Name',y='prop',hue='Recommended IND',data=data['Department Name'].groupby(data['Recommended IND']).value_counts(normalize=True).rename('prop').reset_index())

