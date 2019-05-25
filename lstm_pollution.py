from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import pandas as pd
from datetime import datetime
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#parsing function for date
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
df = pd.read_csv(r'C:\Users\pathak\python\Machine Learning\pollution.csv',  parse_dates = [['year', 'month', 'day', 'hour']],index_col=0, date_parser=parse)
df

#dropping 'No' column as it's of no use
df.drop('No',axis=1)

#naming index as date
df.index.name='date'

#filling null values of pm2.5
df['pm2.5'].fillna(0,inplace=True)

#skipping first 24 rows as they contain null values
df=df[24:].values

#encoding 'cwbd' column as it is non-numeric
encoder = preprocessing.LabelEncoder()
df[:,5] = encoder.fit_transform(df[:,5])

#scaling values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled=scaler.fit_transform(df)
scaled

#dividing dataset into train and test
s=365*24
train=scaled[:s,:]
train
test=scaled[s:,:]

#creating dataset according to the timestamp
def create_dataset(dataset, look_back=1):
  dataX, dataY = [],[]
  for i in range(look_back,len(dataset)):
      dataX.append(dataset[i-look_back:i,:])
      #print("1")
      dataY.append(dataset[i,:])
  return np.array(dataX), np.array(dataY)

train_x,train_y=create_dataset(train)
train_y
test_x,test_y=create_dataset(test)

train_x=train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2])
test_x=test_x.reshape(test_x.shape[0],test_x.shape[1],test_x.shape[2])

#applying lstm model
model = Sequential()
model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(0.2))
#    model.add(LSTM(70))
#    model.add(Dropout(0.3))
model.add(Dense(train_y.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')

# fit network
history = model.fit(train_x, train_y, epochs=200, batch_size=70, validation_data=(test_x, test_y), verbose=2, shuffle=False)

#predicted results
predicted=model.predict(test_x)
test_y
#considering only 4th column(i.e. closing price) for comparison
predicted_new=predicted[:,0:1]
predicted
test_y_new=test_y[:,0:1]
test_y

#visualising predicted vs actual
plt.plot(test_y_new,color="red",label="actual data")
plt.plot(predicted_new,color="blue",label="predicted data")
plt.title("pollution")
plt.xlabel("Time")
plt.ylabel("pm2.5")
plt.legend()
plt.show()