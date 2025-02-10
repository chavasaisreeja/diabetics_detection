import pandas as pd
from sklearn import model_selection
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors 
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
import joblib
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import neural_network
from sklearn.model_selection import train_test_split

root = os.path.dirname(__file__)
path_df = os.path.join(root, 'Diabetics.xlsx')
data = pd.read_excel(path_df)
X= data.drop('Diagnosis',axis=1)
Y=data['Diagnosis']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size=0.1,random_state=110)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clfr = neighbors.KNeighborsClassifier(n_neighbors=78).fit(X_train, y_train)
acc3 = clfr.score(X_test, y_test)
print("Accuracy for KNN: ",acc3*100," %.")

model_path = os.path.join(root, 'models/rfc.sav')
joblib.dump(clfr, model_path)

# Saving the scaler object
scaler_path = os.path.join(root, 'models/scaler.pkl')
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)