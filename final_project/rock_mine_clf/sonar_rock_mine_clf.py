from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from numpy import arange
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import warnings
import pickle
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

dataframe = read_csv("./../../../../datasets/sonar.all-data.csv" )
print(dataframe.shape)
dataset = dataframe 
# print(dataframe.dtypes)
# print(dataframe.head(10))
# print(dataframe.describe)
# print(dataframe.corr())

# dataframe.hist()
# dataframe.plot(kind = 'density',subplots= True,layout = (8,8),sharex = False,sharey = False)
# pyplot.show()
# scatter_matrix(dataframe)
# fig = pyplot.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(dataframe.corr(),vmin=-1,vmax=1,interpolation = 'none')
# fig.colorbar(cax)
# ticks = arange(1,14,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# pyplot.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:60].astype(float)
Y = array[:,60]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)

num_folds = 10
seed = 7
scoring = 'accuracy'



# pipelines = []
# val = [10,20,30,33,35]
# for n in val:
#     for p in val:
#         features = []
#         features.append(('pca', PCA(n_components=p)))
#         features.append(('select_best', SelectKBest(k=n)))
#         feature_union = FeatureUnion(features)
#         pipelines.append(('SVC'+str(p)+'n'+str(n), Pipeline([('Scaler', StandardScaler()),('feature_union',feature_union),('SVM', SVC())])))
# results = []
# names = []
# for name, model in pipelines:
#     kfold = KFold(n_splits=num_folds, random_state=seed)
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     if cv_results.mean()>0.8:
#         msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#         print(msg)

# fig = pyplot.figure()
# fig.suptitle('standardized_Algorithm comparision PCA')
# ax = fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(names)
# pyplot.savefig("standardized__PCA_Algorithm_comparision.png")
# pyplot.show()
# Tune scaled KNN
# Tune scaled SVM
# scaler = StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# rescaledX = PCA(n_components = 33).fit_transform(rescaledX)
# c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
# kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
# param_grid = dict(C=c_values, kernel=kernel_values)
# model = SVC()
# kfold = KFold(n_splits=num_folds, random_state=seed)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
# grid_result = grid.fit(rescaledX, Y_train)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

#training final model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
pca = PCA(n_components = 34).fit(rescaledX)
rescaledX = pca.transform(rescaledX)
model = SVC(C = 1.0,kernel = 'rbf',gamma= 'auto')
model.fit(rescaledX,Y_train)
rescaledXvalidation = scaler.transform(X_validation)
rescaledXvalidation = pca.transform(rescaledXvalidation)
pred = model.predict(rescaledXvalidation)

print('final report')
print('accuracy:',accuracy_score(Y_validation,pred))
print('confusion_matrix:',confusion_matrix(Y_validation,pred))
print('classification report',classification_report(Y_validation,pred))

pickle.dump(model,open("rock_vs_mine_classifier.pkl","wb"))