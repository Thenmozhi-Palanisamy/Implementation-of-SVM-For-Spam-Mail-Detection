# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Thenmozhi
RegisterNumber:  212221230116
*/


import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

*/
````
## Output:

Result output:

![image](https://github.com/Thenmozhi-Palanisamy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/95198708/af2bda09-bd55-43aa-ad08-8a6113f69948)


data.head:


![image](https://github.com/Thenmozhi-Palanisamy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/95198708/71067bff-41ea-47de-b8a1-aeddd0fbd0f5)


data.info():

![image](https://github.com/Thenmozhi-Palanisamy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/95198708/c3c144c5-ccbd-4931-b26e-55d3dc8a8198)

data.isnull().sum()

![image](https://github.com/Thenmozhi-Palanisamy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/95198708/bb359457-8d6f-4eee-8055-a7b0e7ca2d1b)

Y_prediction value:

![image](https://github.com/Thenmozhi-Palanisamy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/95198708/24b57ea1-2b2d-4b76-8677-29f3be9a7c38)

Accuracy value:

![image](https://github.com/Thenmozhi-Palanisamy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/95198708/0f49ff55-966d-44c1-ad31-70845c11190f)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
