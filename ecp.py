import numpy as np
import pandas as pd
import difflib 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
data = pd.read_csv(r'F:\codecatylist\lpk\uploads\EC.csv')
data.shape
data.info()
data.isnull().sum()
x = data.iloc[:, :5]
y = data['Consumption']
print(x)           
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
x_train['Date'] = pd.to_datetime(x_train['Date']).dt.hour
print(x_train)
data.info()

from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy='mean')  
x_train_imputed = imputer.fit_transform(x_train)
x_train = x_train.dropna()
y_train = y_train.dropna()
y_train = y_train.dropna()
x_test['Date']=pd.to_datetime(x_test['Date']).dt.dayofweek
from sklearn.linear_model import LinearRegression


model = LinearRegression()
model.fit(x_train, y_train)
# pickle.dump(model,open('model.pkl','wb'))
# model1=pickle.load(open('model.pkl','rb'))
predictions = model.predict(x_test)
model.fit(x_train, y_train)
model.score(x_test,y_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
plt.bar(y_test, predictions)
plt.xlabel('Actual Electricity Consumption')
plt.ylabel('Predicted Electricity Consumption')
plt.title('Actual vs. Predicted Electricity Consumption')
# plt.show()
plt.savefig('static/ml_graph.png')  
predicted_values = model.predict(x_test)
print(predictions)
predicted_values = model.predict(x_test)
average_prediction = np.mean(predicted_values)

print("Average Predicted Value:", average_prediction)

