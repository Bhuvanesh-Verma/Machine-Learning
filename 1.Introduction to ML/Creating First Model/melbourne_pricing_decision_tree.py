import pandas as pd 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

file_path = "../Datasets/Melbourne_Data_Set/melb_data.csv"
file_data = pd.read_csv(file_path)


#Chossing Dataset for model

y = file_data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = file_data[features]

#Creating training and testing data

train_x , test_x, train_y, test_y = train_test_split(X,y,random_state=0)


model = DecisionTreeRegressor(random_state=2) # defining a model

#fitting model
model.fit(train_x,train_y) 


#predicting values
predicted_values = model.predict(test_x)

actual_values = test_y

print("Predicted values {} \nActual Values {}".format(predicted_values[0:5],actual_values[0:5].tolist()))

#calculating average error

print(mean_absolute_error(test_y,predicted_values))

#lets minimize error by adjusting maximun leaves of tree

for i in [10,100,500,5000]:
    model = DecisionTreeRegressor(max_leaf_nodes=i,random_state=0)
    model.fit(train_x,train_y)
    predicted_values = model.predict(test_x)
    print(mean_absolute_error(test_y,predicted_values))