from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Load your dataset
root = os.path.dirname(__file__)
path_df = os.path.join(root, 'Diabetics.xlsx')
data = pd.read_excel(path_df)
X = data.drop('Diagnosis', axis=1)
Y = data['Diagnosis']

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=12)

# Initialize the TPOTClassifier, which will use genetic programming to find the best model
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=12)

# Fit the TPOTClassifier to the training data
tpot.fit(X_train, y_train)

# Output the accuracy of the model found by TPOT
print("Accuracy for the best model: ", tpot.score(X_test, y_test) * 100, " %")

# Export the best model pipeline
tpot.export(os.path.join(root, 'best_model_pipeline.py'))
