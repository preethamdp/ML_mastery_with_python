from pandas import read_csv
from pandas import set_option
from pandas import to_numeric
from matplotlib import pyplot
import numpy
import pickle
from pandas.plotting import scatter_matrix

data = pickle.load(open("./../pickle_files/diabetes_dataset.pkl","rb"))
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
print(data.head())
#datatypes of each column
print(data.dtypes)
#describing mean sd variance ect of data
print("description")
print(data.describe())
print("class info:")
print(data.groupby('class').size())


data = data.apply(to_numeric)
print(data.dtypes)
correlations = data.corr(method='pearson')
print("correlations:--------")
print(correlations)
print("skew:-----------------")
print(data.skew())
# data.hist()

# data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)

# data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
# pyplot.show()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
scatter_matrix(data)
pyplot.show()