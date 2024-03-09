import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings

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

# Graphs and frequencies
print("no of countries \n", len(df['Area'].unique()))

# displaying a bar graph
plt.figure(figsize=(15,20))
sns.countplot(y=df['Area'])
plt.show()

print("no of countries below 500 crop yield\n", (df['Area'].value_counts() < 500).sum())

country = df['Area'].unique()
yield_per_country = []
for state in country:
    yield_per_country.append(df[df['Area'] == state]['hg/ha_yield'].sum())
print("Yield per country", df['hg/ha_yield'].sum())
# print("Yield per country sum",yield_per_country)
# Yield per country bar-graph
plt.figure(figsize=(15, 20))
plt.title('Yield per country\n')
sns.barplot(y=country, x=yield_per_country)
plt.show()

# countplot
sns.countplot(y=df['Item'])
plt.show()

# Crop yield vs Item
crops = df['Item'].unique()
yield_per_crop = []
for crop in crops:
    yield_per_crop.append(df[df['Item'] == crop]['hg/ha_yield'].sum())
sns.barplot(y=crops, x=yield_per_crop)
plt.show()

# Train-Test-Split
col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(df.head(3))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)

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
    'Dtr': DecisionTreeRegressor()
}

for name, md in models.items():
    md.fit(X_train_dummy, y_train)
    y_pred = md.predict(X_test_dummy)

    if name == 'lr':  # For Linear Regression
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} : MAE : {mae} R-squared : {r2}")
    else:  # For other algorithms
        print(f"{name} : MAE : {mean_absolute_error(y_test, y_pred)} R-squared : {r2_score(y_test, y_pred)}")
    
    
dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy, y_train)

# Make predictions
y_pred = dtr.predict(X_test_dummy)


# Create a scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs. Predicted Crop Yield - Decision Tree")
plt.show()



def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    # Create an array of the input features
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

    # Transform the features using the preprocessor
    transformed_features = preprocesser.transform(features)

    # Make the prediction
    predicted_yield = dtr.predict(transformed_features).reshape(1, -1)

    return predicted_yield[0]




pickle.dump(dtr, open('dtr.pkl', 'wb'))
pickle.dump(preprocesser, open('preprocessor.pkl', 'wb'))

warnings.simplefilter(action='ignore', category=FutureWarning)
