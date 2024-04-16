from math import ceil
import numpy as np
import pandas as pd
from scipy import linalg
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
df = pd.read_csv('E:\\VEC\\Crop_Yield_Prediction\\yield_df1.csv')

# Drop 'Unnamed' column if present
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Handling Null values
print("Handling Null values")
print("Null:", df.isnull().sum())

# Dropping Duplicate Entries
print("Duplicate Entries")
print("Duplicates:", df.duplicated().sum())

print("Dropping Duplicates")
df.drop_duplicates(inplace=True)
print(df.head())

# Function to check if a value is a string
def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True

# Drop rows with non-numeric values in 'average_rain_fall_mm_per_year' column
to_drop = df[df['average_rain_fall_mm_per_year'].apply(isStr)].index
df.drop(to_drop, inplace=True)

print("Converting column 3 to Numerics")
print("converting average rainfall column to float")
df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(np.float64)

# Visualizing top 50 Areas
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

# Visualizing Crop Yield vs Item
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing: Converting Categorical to Numerical and Scaling the values
ohe = OneHotEncoder(drop='first')
scale = MinMaxScaler()

preprocesser = ColumnTransformer(  transformers=[
        ('StandardScale', scale, [0, 1, 2, 3]),
        ('OHE', ohe, [4, 5]),
    ],
    remainder='passthrough'
)

X_train_dummy = preprocesser.fit_transform(X_train)
X_test_dummy = preprocesser.transform(X_test)

# Define the number of folds for cross-validation
num_folds = 10  # You can adjust this number as needed

# Initialize KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Training the model
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(),
    'Ridge Regression': Ridge(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Support Vector Regressor' : LinearSVR(),
    'K Nearest Neighbours Regressor' : KNeighborsRegressor(n_neighbors=100),
    'Gradient Boosting Regressor' : GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
}

plt.figure(figsize=(12, 10))

for name, md in models.items():
    scores = cross_val_score(md, X_train_dummy, y_train, cv=10, scoring='neg_mean_absolute_error')
    avg_mae = -scores.mean()
    scores_r2 = cross_val_score(md, X_train_dummy, y_train, cv=10, scoring='r2')
    avg_r2 = scores_r2.mean()
    print('After applying Cross Validation\n')
    print(f"{name} : \n Mean-Absolute-Error(MAE)  : {avg_mae} \t Mean R-squared : {avg_r2} \n\n")
    
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
y_pred = knn.predict(X_test_dummy)

# Plot Actual vs Predicted Crop Yield
sns.scatterplot(x=y_test, y=y_pred_dtr, label='hg/ha yield', alpha=0.7, s=80)

# Add a diagonal line for comparison
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray')

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.legend()
plt.show()  

# Function to make predictions
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    # Create an array of the input features
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

    # Transform the features using the preprocessor
    transformed_features = preprocesser.transform(features)

    # Make the prediction
    predicted_yield = knn.predict(transformed_features).reshape(1, -1)

    return predicted_yield[0]

# Serialize models and preprocesser
pickle.dump(dtr, open('dtr.pkl', 'wb'))
pickle.dump(preprocesser, open('preprocessor.pkl', 'wb'))
