from pandas import read_csv
from pandas import set_option
from pandas import to_numeric
from matplotlib import pyplot
import numpy
import pickle
filename = './../../../datasets/pima-indians-diabetes-database/diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)
set_option('display.width', 100)
set_option('precision', 3)
data.drop([0],inplace = True)
print(data.shape)

pickle.dump(data,open("./../pickle_files/diabetes_dataset.pkl","wb"))


#loading housing dataset
filename = './../../../datasets/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
data = read_csv(filename, delim_whitespace=True, names=names)
print(data.head())
pickle.dump(data,open("./../pickle_files/housing_dataset.pkl","wb"))