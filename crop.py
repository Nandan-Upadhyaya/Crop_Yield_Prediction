from math import ceil
import numpy as np
import pandas as pd
from scipy import linalg
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#File path
df = pd.read_csv('C:\\Users\\Nandan Upadhyaya.DESKTOP-CKL8RDH\\Desktop\\Machine Learning\\Crop_Yield_Prediction\\yield_df1.csv')

#Dropping Unnamed Column
# Dropping Unnamed Column
df.drop('Unnamed: 0', axis=1, inplace=True)

print("Handling Null values")
print("Null:", df.isnull().sum())

print("Duplicate Entries")
print("Duplicates:", df.duplicated().sum())

print("Dropping Duplicates")
df.drop_duplicates(inplace=True)
print(df.head())

def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True

to_drop = df[df['average_rain_fall_mm_per_year'].apply(isStr)].index
df.drop(to_drop, inplace=True)

print("Converting column 3 to Numerics")
print("converting average rainfall column to float")
df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(np.float64)

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

def lowess(x, y, f, iterations):
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iterations):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]
        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    return yest

# Train-Test-Split
col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(df.head(3))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_lowess = X_train['avg_temp'].values.reshape(-1, 1)
y_train_lowess = y_train.values
yest_lowess = lowess(X_train_lowess.ravel(), y_train_lowess, f=0.90, iterations=20)
r2_lowess = r2_score(y_train_lowess, yest_lowess)
print(f"LOWESS R-squared: {r2_lowess}")

# Converting Categorical to Numerical and Scaling the values
ohe = OneHotEncoder(drop='first')
scale = MinMaxScaler()

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
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(),
    'Ridge Regression': Ridge(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Support Vector Regressor' : LinearSVR() ,
    'K Nearest Neighbours Regressor' : KNeighborsRegressor(n_neighbors=100)
}

plt.figure(figsize=(12, 10))

for name, md in models.items():
    md.fit(X_train_dummy, y_train)
    y_pred = md.predict(X_test_dummy)

    if name == 'Linear Regression':  # For Linear Regression
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test,y_pred)
        print(f"{name} : \n Mean-Absolute-Error(MAE) : {mae} \t R-squared : {r2} \t Mean-Squared-Error(MSE) : {mse} \n\n")
    else:  # For other algorithms
        print(f"{name} : \n  Mean-Absolute-Error(MAE) : {mean_absolute_error(y_test, y_pred)} \t R-squared : {r2_score(y_test, y_pred)} \t Mean-Squared-Error(MSE)  : {mean_squared_error(y_test, y_pred)}\n\n ")
    
    
knn = models['K Nearest Neighbours Regressor']
dtr = models['Decision Tree Regressor']
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

plt.figure(figsize=(8, 6))

plt.scatter(X_train_lowess, y_train_lowess, color='blue', label='Actual')
plt.scatter(X_train_lowess, yest_lowess, color='red', label='Predicted')
plt.plot([min(y_train_lowess), max(y_train_lowess)], [min(y_train_lowess), max(y_train_lowess)], color='gray', linestyle='--', label='45° line')

plt.ylabel('Crop Yield (hg/ha)')
plt.title('Actual vs Predicted Crop Yield using LWR')
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


