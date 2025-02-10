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
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size=0.2,random_state=30)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf2 = neural_network.MLPClassifier(activation='relu',
    solver='lbfgs',
    early_stopping=False,
    hidden_layer_sizes=(8,4,4,4,2),
    random_state=100,
    batch_size='auto',
    max_iter=10000,  # 94.81
   learning_rate_init=1e-5,
    tol=1e-4).fit(X_train, y_train)

acc1 = clf2.score(X_test, y_test)
print("Accuracy for MLP: ",acc1*100," %.")

scaler_path = os.path.join(root, 'models/scaler.pkl')
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)