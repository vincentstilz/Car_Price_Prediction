# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# IMPORTING THE DATA SET AND CHECK IT
file_path = "/Users/vincent/Desktop/Python/CarPricePred/car_prices.csv"
car_data = pd.read_csv(file_path)
car_data.head()
car_data.info()
car_data.describe().round()

# DATA CLEANING
#dropping missing values 
car_data.dropna(inplace=True)
car_data.isna().sum()

# SUPERVISED ALGORITHM
#defining the dependent and the indipendent varible for a regression 
y = car_data['sellingprice']

feature_columns = ['year', 'make', 'model', 'odometer', 'condition']
X = car_data[feature_columns] 

#spliting the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#preprocessing the data
categorical_features = ['make', 'model']
numerical_features = ['year', 'odometer', 'condition']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

#define the model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

#fit the model and test it
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred[:5])  
print(f'Linear Regression MSE: {mean_squared_error(y_test, y_pred)}')
print(f'Linear Regression R² score: {r2_score(y_test, y_pred)}')
# The model's predictions have an average squared error of 41.1 million, indicating substantial deviations from actual selling prices.
# With an R² score of 0.547, the model explains about 54.7% of the variance in car selling prices from the selected features.

#update the model pipeline to use a random forrest regressor 
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print(y_pred[:5]) 
print(f'Random Forest Regression MSE: {mean_squared_error(y_test, y_pred)}')
print(f'Random Forest Regression R² score: {r2_score(y_test, y_pred)}')
# The Random Forest model significantly improves prediction accuracy with an MSE of ~2.89 million and explains ~96.82% of the variance in car selling prices (R² score).
# Predicted selling prices for the first five cars in the test set range from approximately $4,049 to $22,966, indicating varied price predictions across the dataset.

# UNSUPERVISED ALGORTHM 
#creating a KMeans clustering algorithm to cluster cars based on their features excluding the price
y = car_data['sellingprice']
X = car_data.drop('sellingprice', axis=1)

#preprocess data based on the data type
categorical_features = ['make', 'model']
numerical_features = ['year', 'odometer', 'condition']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Elbow Method to determin number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('cluster', kmeans)])
    pipeline.fit(X)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

#final model fit
pipeline.fit(X)
# optimal_k determined based on Elbow method
optimal_k = 4  
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('cluster', kmeans)])
#visualise the clusters in a 2D space using dimensionality reduction
X = car_data.drop('sellingprice', axis=1)

#preprocess the data based on data type
categorical_features = ['make', 'model']
numerical_features = ['year', 'odometer', 'condition']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

#define the clustering pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=3, random_state=42))
])

#fit the model
pipeline.fit(X)

#dimensionality reduction 
svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(pipeline.named_steps['preprocessor'].transform(X))

#plotting the clusters
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=pipeline.named_steps['cluster'].labels_, cmap='viridis')
plt.title('2D Visualization of Car Data Clusters')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Cluster Label')
plt.show()