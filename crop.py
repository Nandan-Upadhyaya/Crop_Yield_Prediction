import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#File path
df = pd.read_csv('C:\\Users\\Nandan Upadhyaya.DESKTOP-CKL8RDH\\Desktop\\Machine Learning\\Crop_Yield_Prediction\\yield_df1.csv')

#Dropping Unnamed Column
df.drop('Unnamed: 0', axis=1, inplace=True)
#print("head:", df.head())

'''print("shape: ", df.shape)

print("Handling Null values")
print("Null:", df.isnull().sum())

print("Duplicate Entries")
print("Duplicates:", df.duplicated().sum())

print("Dropping Duplicates")
df.drop_duplicates(inplace=True)
print(df.head())

print("Describing")
print(df.describe())'''

def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True
    
to_drop = df[df['average_rain_fall_mm_per_year'].apply(isStr)].index
df.drop(to_drop, inplace=True)

print("Converting column 3 to Numerics")
#print(df.head())

print("converting average rainfall column to float")
df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(np.float64)
#print(df.head())

plt.figure(figsize=(15, 10))
top_areas = df['Area'].value_counts().nlargest(50).index
sns.countplot(y=df[df['Area'].isin(top_areas)]['Area'])
plt.title('Count of Top 50 Areas')
plt.xlabel('Count')
plt.ylabel('Area')
plt.show()


# Identify the top 50 areas with the highest crop yield
top_areas = df.groupby('Area')['hg/ha_yield'].sum().sort_values(ascending=False).head(50).index

# Filter the dataframe for the top 50 areas
df_top_areas = df[df['Area'].isin(top_areas)]

# Yield per country bar-graph for the top 50 areas
#Graph is not displaying in the right way
'''plt.figure(figsize=(15, 10))
plt.title('Yield per country bar graph(Top 50) \n')
sns.barplot(y='Area', x='hg/ha_yield', data=df_top_areas, order=top_areas)
plt.show()
# countplot
sns.countplot(y=df['Item'])
plt.show()'''

# Crop yield vs Item
crops = df['Item'].unique()
yield_per_crop = []
for crop in crops:
    yield_per_crop.append(df[df['Item'] == crop]['hg/ha_yield'].sum())
sns.barplot(y=crops, x=yield_per_crop)
plt.title('Crop Yield vs Item')
plt.show()

# Train-Test-Split
col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(df.head(3))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=True)

# Converting Categorical to Numerical and Scaling the values
ohe = OneHotEncoder(drop='first')
scale = StandardScaler()

preprocesser = ColumnTransformer(
    transformers=[
        ('StandardScale', scale, [0, 1, 2, 3]),
        ('OHE', ohe, [4, 5]),
    ],
    remainder='passthrough'
)

X_train_dummy = preprocesser.fit_transform(X_train)
X_test_dummy = preprocesser.transform(X_test)
'''print(preprocesser.get_feature_names_out(col[:-1]))'''

# Training the model
models = {
    'lr': LinearRegression(),
    'lss': Lasso(),
    'Rid': Ridge(),
    'Dtr': DecisionTreeRegressor(),
    'knn' : KNeighborsRegressor(n_neighbors=100)
}

plt.figure(figsize=(12, 10))

for name, md in models.items():
    md.fit(X_train_dummy, y_train)
    y_pred = md.predict(X_test_dummy)

    if name == 'lr':  # For Linear Regression
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test,y_pred)
        print(f"{name} : MAE : {mae} R-squared : {r2} MSE : {mse}")
    else:  # For other algorithms
        print(f"{name} : MAE : {mean_absolute_error(y_test, y_pred)} R-squared : {r2_score(y_test, y_pred)} MSE : {mean_squared_error(y_test, y_pred)} ")
    
    
knn = models['knn']
dtr = models['Dtr']
y_pred_knn = knn.predict(X_test_dummy)
y_pred_dtr = dtr.predict(X_test_dummy)


# Make predictions
y_pred =knn.predict(X_test_dummy)



sns.scatterplot(x=y_test, y=y_pred, label='hg/ha yield', alpha=0.7, s=80)

# Add a diagonal line for comparison
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray')

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.legend()
plt.show()

def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    # Create an array of the input features
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

    # Transform the features using the preprocessor
    transformed_features = preprocesser.transform(features)

    # Make the prediction
    predicted_yield = knn.predict(transformed_features).reshape(1, -1)

    return predicted_yield[0]




pickle.dump(dtr, open('dtr.pkl', 'wb'))
pickle.dump(preprocesser, open('preprocessor.pkl', 'wb'))


