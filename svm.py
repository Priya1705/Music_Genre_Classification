import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score
     
dataset= np.genfromtxt("data_2genre.csv", delimiter=",")

dataset= dataset[1:,]
data=dataset[0:,1:]

df=pd.DataFrame(data)
df.to_csv( "data.csv",index=False, header=False)

fixdata=pd.read_csv('data.csv')

x=fixdata.iloc[:,0:-1]
y=fixdata.iloc[:,-1]

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)

support = svm.LinearSVC(random_state=20)

support.fit(x_train, y_train)
predicted= support.predict(x_test)
score=accuracy_score(y_test,predicted)
print("Your Model Accuracy is", score)

