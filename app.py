#https://stackoverflow.com/questions/46785507/python-flask-display-image-on-a-html-page
#https://github.com/HarunHM/Login-System-with-Python-Flask-and-MySQL/blob/master/main.py
#pip install scikit-learn==0.23.2
from flask import Flask, render_template, url_for, request
import joblib
import os
import numpy as np
import pickle

#mysql details
'''from flask_mysqldb import MySQL
app = Flask(__name__)


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 't2d'

mysql = MySQL(app)'''


#over


app = Flask(__name__, static_folder='static')

PEOPLE_FOLDER = os.path.join('static', 'people_photo')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route("/")
def index1():
	app = Flask(__name__, static_folder='static')

	PEOPLE_FOLDER = os.path.join('static', 'people_photo')
	app = Flask(__name__)
	app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
	PEOPLE_FOLDER = os.path.join('static', 'people_photo')
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 't2dinsert1.jpeg')
	#cur = mysql.connection.cursor()
	return render_template('home.html', user_image = full_filename)


@app.route("/home")
def index():
	PEOPLE_FOLDER = os.path.join('static', 'people_photo')
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 't2dinsert1.jpeg')
	return render_template('home.html', user_image = full_filename)
	
	
	
@app.route("/main")
def home():
	PEOPLE_FOLDER = os.path.join('static', 'people_photo')
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 't2dinsert1.jpeg')
	return render_template('main.html', user_image = full_filename)
	
	
@app.route('/result', methods=['POST', 'GET'])
def result():
	Pregnant = int(request.form['Number of times pregnant'])
	Pg = int(request.form['Plasma glucose concentration a 2 hours in an oral glucose tolerance test'])
	DB = float(request.form['Diastolic blood pressure (mm Hg)'])
	Triceps = float(request.form['Triceps skin fold thickness (mm)'])
	SI = float(request.form['2-Hour serum insulin (mu U/ml)'])
	BMI = float(request.form['Body mass index (weight in kg/(height in m)^2)'])
	DPF = float(request.form['Diabetes pedigree function'])
	age = int(request.form['Age (years)'])
	
	x = np.array([Pregnant, Pg, DB,	  Triceps,	SI, BMI, DPF , age]).reshape(1, -1)

	scaler_path = os.path.join(os.path.dirname(__file__), 'models/scaler.pkl')
	scaler = None
	with open(scaler_path, 'rb') as f:
		scaler = pickle.load(f)

	x = scaler.transform(x)

	model_path = os.path.join(os.path.dirname(__file__), 'models/rfc.sav')
	clf = joblib.load(model_path)

	y = clf.predict(x)
	print(y)

	if y == 0:
		return render_template('nodisease.html')

  
	else:
		return render_template('diabeticdisease.html', stage=int(y))


@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/decisiontree')
def decisiontree():
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
	X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size=0.2,random_state=95)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
	'''print(data[(data['Diagnosis']=='0')].value_counts())
	print(data[(data['Diagnosis']=='1')].value_counts())'''
	data = data['Diagnosis'].value_counts().reset_index()
	data.columns = ['Diagnosis', 'count']	
	print(data)
	clfr = DecisionTreeClassifier(criterion = 'entropy', random_state = 11).fit(X_train, y_train)
	acc4 = clfr.score(X_test, y_test)
	d = {'a': acc4*100}
	
	print("Accuracy for Decision tree: ",acc4*100," %.")
	
	model_path = os.path.join(root, 'models/rfc.sav')
	joblib.dump(clfr, model_path)
	
	# Saving the scaler object
	scaler_path = os.path.join(root, 'models/scaler.pkl')
	'''with open(scaler_path, 'wb') as scaler_file:
		pickle.dump(scaler, scaler_file)
		return render_template('about.html')''' 
	return render_template('accdecision.html',data =d)	
@app.route('/knn')
def knn():
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
	d = {'a': acc3*100}
	print("Accuracy for KNN: ",acc3*100," %.")
	

	model_path = os.path.join(root, 'models/rfc.sav')
	joblib.dump(clfr, model_path)

	# Saving the scaler object
	scaler_path = os.path.join(root, 'models/scaler.pkl')
	'''with open(scaler_path, 'wb') as scaler_file:
		pickle.dump(scaler, scaler_file)
		return render_template('about.html')'''
	return render_template('accknn.html',data =d)
@app.route('/mlp')
def mlp():
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
		max_iter=10000,	 # 94.81
	   learning_rate_init=1e-5,
		tol=1e-4).fit(X_train, y_train)

	acc1 = clf2.score(X_test, y_test)
	d = {'a': acc1*100}
	print("Accuracy for MLP: ",acc1*100," %.")
	

	'''scaler_path = os.path.join(root, 'models/scaler.pkl')
	with open(scaler_path, 'wb') as scaler_file:
		pickle.dump(scaler, scaler_file)
		return render_template('about.html')'''
	return render_template('accmlp.html',data =d)
@app.route('/randomforest')
def randomforest():
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
	X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size=0.05,random_state=0)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	clf1 = ensemble.RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0).fit(X_train, y_train)
	acc = clf1.score(X_test, y_test)
	d = {'a': acc*100}
	print("Accuracy for random forest : ",acc*100," %.")

	model_path = os.path.join(root, 'models/rfc.sav')
	joblib.dump(clf1, model_path)
	# Saving the scaler object	
	return render_template('accrandomforest.html',data =d)
@app.route('/svm')
def svm():
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
	X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size=0.1,random_state=12)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	cl = svm.SVC(kernel = 'linear',C = 0.1).fit(X_train, y_train)
	accsvm = cl.score(X_test, y_test)
	d = {'a': accsvm*100}
	print("Accuracy for svm: ",accsvm*100," %.")


	model_path = os.path.join(root, 'models/rfc.sav')
	joblib.dump(cl, model_path)

	# Saving the scaler object
	'''scaler_path = os.path.join(root, 'models/scaler.pkl')
	with open(scaler_path, 'wb') as scaler_file:
		pickle.dump(scaler, scaler_file)
		return render_template('about.html')'''
	return render_template('accsvm.html',data =d)	
@app.route('/graph')
def graph():
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
	X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size=0.2,random_state=95)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	#decision tree
	clfr = DecisionTreeClassifier(criterion = 'entropy', random_state = 11).fit(X_train, y_train)
	acc4 = clfr.score(X_test, y_test)
	d = {'a': acc4*100}
	print("Accuracy for Decision tree: ",acc4*100," %.")
	
	
	#random forest
	clf1 = ensemble.RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0).fit(X_train, y_train)
	acc = clf1.score(X_test, y_test)
	d = {'a': acc*100}
	print("Accuracy for random forest : ",acc*100," %.")
	
	
	#mlp
	clf2 = neural_network.MLPClassifier(activation='relu',
	solver='lbfgs',
	early_stopping=False,
	hidden_layer_sizes=(8,4,4,4,2),
	random_state=100,
	batch_size='auto',
	max_iter=10000,	 # 94.81
	learning_rate_init=1e-5,
	tol=1e-4).fit(X_train, y_train)

	acc1 = clf2.score(X_test, y_test)
	d = {'a': acc1*100}	 

	#knn
	clfr = neighbors.KNeighborsClassifier(n_neighbors=78).fit(X_train, y_train)
	acc3 = clfr.score(X_test, y_test)
	d = {'a': acc3*100}
	
	#svm
	cl = svm.SVC(kernel = 'linear',C = 0.1).fit(X_train, y_train)
	accsvc = cl.score(X_test, y_test)
	
	
	print("Accuracy for GA: ",accsvc*100," %.")
	print("Accuracy for KNN: ",acc3*100," %.")
	meanse=[acc4,acc,acc1,acc3,accsvc]
	#bargraph for meansquarederror
	import matplotlib.pyplot as plt
	fig = plt.figure()

	ax = fig.add_axes([0,0,1,1])
	ax.axis('equal')
	langs = ['Decsntree','RandomForest','mlp','knn','GA']
	students = [meanse[0],meanse[1],meanse[2],meanse[3],meanse[4]]
	#ax.pie(langs,students)
	ax.pie(students, labels = langs,autopct='%1.2f%%')

	plt.show()
	
	
	dict={'dt':acc4*100,'mlp':acc1*100,'knc':acc3*100,'GA':accsvc*100}
	
	################

	#####################
	model_path = os.path.join(root, 'models/rfc.sav')
	joblib.dump(clfr, model_path)
	
	app = Flask(__name__, static_folder='static')

	PEOPLE_FOLDER = os.path.join('static', 'people_photo')
	app = Flask(__name__)
	app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
	PEOPLE_FOLDER = os.path.join('static', 'people_photo')
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 't2dinsert1.jpeg')
	#cur = mysql.connection.cursor()
	return render_template('index2.html', user_image = full_filename,data=dict)
if __name__ == "__main__":
	app.run(debug=True, port=5000)