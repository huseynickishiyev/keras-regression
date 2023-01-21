import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv(r'C:\Users\USR\Desktop\ML\kc_house_data.csv')

df

df.describe()
df.isnull().sum()

df = df.drop(['date', 'id', 'zipcode'], axis=1)

def plotHistogram(variable):
    """
    Parameters
    ----------
    variable : numerical variable.
    
    Returns
    -------
    Histogram.
    """
    plt.figure(figsize=(10,5))
    plt.hist(df[variable], bins=85, color="blue")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Data Frequency - {variable}")
    plt.show()
    

numerical_variables = ['bedrooms',	'bathrooms',	'sqft_living',	'sqft_lot',	'floors']

plot_ = [plotHistogram(i) for i in numerical_variables]

plot_    

from sklearn.model_selection import train_test_split

X = df.drop(['price'], axis=1).values
y = df['price'].values    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

X_train.shape

model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))


model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(x=X_train, y=y_train, 
          validation_data=(X_test, y_test),
          batch_size=128, epochs=400)

losses = pd.DataFrame(model.history.history)
losses

losses.head(50)

plt.figure(figize=(10,5))
losses.plot()

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

predictions = model.predict(X_test)
predictions

mean_squared_error(y_test, predictions)

mean_absolute_error(y_test, predictions)

    

    
