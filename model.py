import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#df=pd.read_csv("pizza_v2.csv")
df=pd.read_csv("D:\\ML model\\pizza_v2.csv")
print(df.head())


df['price_rupiah'] = df['price_rupiah'].astype(str)
df['diameter']=df['diameter'].str.replace('inch','').astype(float)
df['price_rupiah'] = df['price_rupiah'].str.replace('Rp', '').str.replace(',', '').astype(int)

# Cat_col=df.select_dtypes(include=['object']).columns
# # importing the Necessary libraries for converting  the object variable into Numerical variables
# from sklearn.preprocessing import LabelEncoder
# # creating an instance of Label encoder() and doing the fit and transformation through a for loop
# en= LabelEncoder()
# for i in Cat_col:
#     df[i]=en.fit_transform (df[i])
# sns.heatmap(df.corr(),annot=True)

df['topping']=df['topping'].replace(to_replace=['chicken', 'papperoni', 'mushrooms', 'smoked_beef', 'mozzarella',
       'black_papper', 'tuna', 'meat', 'sausage', 'onion', 'vegetables',
       'beef'],value=['0','1','2','3','4','5','6','7','8','9','10','11']).astype(int)

df['company']=df['company'].replace(to_replace=['A', 'B', 'C', 'D', 'E'],value=['0','1','2','3','4']).astype(int)
df['variant']=df['variant'].replace(to_replace=['double_signature', 'american_favorite', 'super_supreme',
       'meat_lovers', 'double_mix', 'classic', 'crunchy', 'new_york',
       'double_decker', 'spicy_tuna', 'BBQ_meat_fiesta', 'BBQ_sausage',
       'extravaganza', 'meat_eater', 'gournet_greek', 'italian_veggie',
       'thai_veggie', 'american_classic', 'neptune_tuna', 'spicy tuna'],value=['0','1','2','3','4','5','6','7','8','9',
        '10','11','12','13','14','15','16','17','18','19']).astype(int)
df['size']=df['size'].replace(to_replace=['jumbo', 'reguler', 'small', 'medium', 'large', 'XL'],value=['0','1','2','3','4',
        '5']).astype(int)
df['extra_sauce']=df['extra_sauce'].replace(to_replace=['yes', 'no'],value=['0','1']).astype(int)
df['extra_cheese']=df['extra_cheese'].replace(to_replace=['yes', 'no'],value=['0','1']).astype(int)
df['extra_mushrooms']=df['extra_mushrooms'].replace(to_replace=['yes', 'no'],value=['0','1']).astype(int)

#select independent and dependent variables
x = df.drop(["price_rupiah"], axis=1)
y = df["price_rupiah"]

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x, y , test_size=0.2)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train , y_train)

pickle.dump(lr,open("model.pkl","wb"))