from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pickle
data = pickle.load(open("./../pickle_files/diabetes_dataset.pkl","rb"))
array = data.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
model = SVC()
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
