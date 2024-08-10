import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics, linear_model, __all__, ensemble
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
import datetime
from sklearn.linear_model import LogisticRegression
import json
from xgboost import XGBClassifier
from sklearn.svm import SVR

data = pd.read_csv('movies-classification-dataset.csv')

X = data.iloc[:, 0:19]
Y = data['Rate']

# split the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)

x_train['release_date'] = pd.to_datetime(x_train['release_date'].values).year


label_encoder = preprocessing.LabelEncoder()
labels_encoding = ['homepage', 'tagline', 'status', 'title', 'original_title', 'original_language']
for label in labels_encoding:
    x_train[label] = label_encoder.fit_transform(x_train[label])

x_train.runtime.replace(to_replace = 0, value = x_train.runtime.mean(), inplace=True)
x_train.budget.replace(to_replace = 0, value = x_train.budget.mean(), inplace=True)

# changing the genres column from json to string
x_train.genres = x_train.genres.apply(json.loads)
for index, i in zip(x_train.genres.index, x_train.genres):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    x_train.loc[index, 'genres'] = str(list1)


for i, j in zip(x_train.genres, x_train.index):
        list2 = []
        list2 = i
        x_train.loc[j, 'genres'] = str(list2)
x_train.genres = x_train.genres.str.strip('[]').str.replace(' ', '').str.replace("'", '')
x_train.genres = x_train.genres.str.split(',')

genreList = []
for index, row in x_train.iterrows():
    genres = row["genres"]

    for genre in genres:
        if genre not in genreList:
            genreList.append(genre)

def derivesColumns(data,genre,column):
  List=[]
  for index, row in data.iterrows():
     genres = row.loc[column]

     if genre not in genres:
        List.append(0)
     else:
        List.append(1)


  data[genre] = List
  return data

for i in genreList :
    x_train= derivesColumns(x_train,i,'genres')
#print(x_train.genres)
#print(data.corr)
# changing the keywords column from json to string
x_train.keywords = x_train.keywords.apply(json.loads)
for index, i in zip(x_train.keywords.index, x_train.keywords):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    x_train.loc[index, 'keywords'] = str(list1)

for i, j in zip(x_train.keywords, x_train.index):
        list3 = []
        list3 = i
        x_train.loc[j, 'keywords'] = str(list3)
x_train.keywords = x_train.keywords.str.strip('[]').str.replace(' ', '').str.replace("'", '')
x_train.keywords = x_train.keywords.str.split(',')

keywordsList = []
for index, row in x_train.iterrows():
    keywords = row["keywords"]

    for keyword in keywords:
        if keyword != '' :
            keywordsList.append(keyword)
keywordsList=pd.Series(keywordsList)
for i in keywordsList.value_counts().keys().tolist()[:10] :
    x_train= derivesColumns(x_train,i,'keywords')

# changing the production_companies column from json to string
x_train.production_companies = x_train.production_companies.apply(json.loads)
for index, i in zip(x_train.production_companies.index, x_train.production_companies):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    x_train.loc[index, 'production_companies'] = str(list1)

for i, j in zip(x_train.production_companies, x_train.index):
        list4 = []
        list4 = i
        x_train.loc[j, 'production_companies'] = str(list4)
x_train.production_companies = x_train.production_companies.str.strip('[]').str.replace(' ', '').str.replace("'", '')
x_train.production_companies = x_train.production_companies.str.split(',')

production_companiesList = []
for index, row in x_train.iterrows():
    production_companies = row["production_companies"]

    for production_company in production_companies:
        if production_company != '':
            production_companiesList.append(production_company)
production_companiesList=pd.Series(production_companiesList)
for i in production_companiesList.value_counts().keys().tolist()[:10] :
    x_train= derivesColumns(x_train,i,'production_companies')

# changing the production_countries column from json to string
x_train.production_countries = x_train.production_countries.apply(json.loads)
for index, i in zip(x_train.production_countries.index,x_train.production_countries):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    x_train.loc[index, 'production_countries'] = str(list1)

for i, j in zip(x_train.production_countries, x_train.index):
        list5 = []
        list5 = i
        x_train.loc[j, 'production_countries'] = str(list5)
x_train.production_countries = x_train.production_countries.str.strip('[]').str.replace(' ', '').str.replace("'", '')
x_train.production_countries = x_train.production_countries.str.split(',')

production_countriesList = []
for index, row in x_train.iterrows():
    production_countries = row["production_countries"]

    for production_country in production_countries:
        if production_country not in production_countriesList:
            production_countriesList.append(production_country)
for i in production_countriesList :
    x_train= derivesColumns(x_train,i,'production_countries')

# changing the spoken_languages column from json to string
x_train.spoken_languages = x_train.spoken_languages.apply(json.loads)
for index, i in zip(x_train.spoken_languages.index, x_train.spoken_languages):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    x_train.loc[index, 'spoken_languages'] = str(list1)

for i, j in zip(x_train.spoken_languages, x_train.index):
        list6 = []
        list6 = i
        x_train.loc[j, 'spoken_languages'] = str(list6)
x_train.spoken_languages = x_train.spoken_languages.str.strip('[]').str.replace(' ', '').str.replace("'", '')
x_train.spoken_languages = x_train.spoken_languages.str.split(',')


spoken_languagesList = []
for index, row in x_train.iterrows():
    spoken_languages = row["spoken_languages"]

    for spoken_language in spoken_languages:
        if spoken_language not in spoken_languagesList:
            spoken_languagesList.append(spoken_language)
for i in spoken_languagesList :
    x_train= derivesColumns(x_train,i,'spoken_languages')


lbl = LabelEncoder()
def Feature_Encoder(X,cols):
    for c in cols:
        X[c]=X[c].astype(str)
        lbl.fit((X[c]))
        X[c] = lbl.transform((X[c]))
    return X

columns=['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']
x_train =Feature_Encoder(x_train,columns)

#print(x_train['production_companies'])
x_train['runtime'].fillna(method='ffill', inplace=True)

# preprocessing on revenue column
Z = x_train.loc[:, 'revenue']
scaler = MinMaxScaler(feature_range=(0, 100))
Z = Z.values.reshape(-1, 1)
standarized_revenue = scaler.fit_transform(Z)
x_train['revenue'] = standarized_revenue

# preprocessing on vote_count column
W = x_train.loc[:, 'vote_count']
scaler = MinMaxScaler(feature_range=(0, 100))
W = W.values.reshape(-1, 1)
standarized_vote_count = scaler.fit_transform(W)
x_train['vote_count'] = standarized_vote_count

# preprocessing on runtime column
F = x_train.loc[:, 'runtime']
scaler = MinMaxScaler(feature_range=(1, 100))
F = F.values.reshape(-1, 1)
standarized_runtime = scaler.fit_transform(F)
x_train['runtime'] = standarized_runtime
#print(standarized_runtime)
#print(data['runtime'])

# preprocessing on viewercount column
K = x_train.loc[:, 'viewercount']
scaler = MinMaxScaler(feature_range=(0, 100))
K = K.values.reshape(-1, 1)
standarized_viewer_count = scaler.fit_transform(K)
x_train['viewercount'] = standarized_viewer_count


rate_map = {'Low': 0, 'Intermediate': 1, 'High': 2}
y_train= y_train.map(rate_map)
rate_map = {'Low': 0, 'Intermediate': 1, 'High': 2}
y_test= y_test.map(rate_map)

data = x_train
data['Rate'] = y_train
corr = data.corr()
top_features = corr.index[abs(corr['Rate']) >  0.09]
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_features.delete(-1)
x_train = data[top_features.delete(-1)]


rf= RandomForestClassifier()
rf.fit(x_train,y_train)

reg = LogisticRegression()
reg.fit(x_train, y_train)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
######################################################################test
x_test.genres = x_test.genres.apply(json.loads)
for index, i in zip(x_test.genres.index, x_test.genres):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    x_test.loc[index, 'genres'] = str(list1)

for i, j in zip(x_test.genres, x_test.index):
        list2 = []
        list2 = i
        x_test.loc[j, 'genres'] = str(list2)
x_test.genres = x_test.genres.str.strip('[]').str.replace(' ', '').str.replace("'", '')
x_test.genres = x_test.genres.str.split(',')

for i in genreList :
    x_test= derivesColumns(x_test,i,'genres')


# changing the keywords column from json to string
x_test.keywords = x_test.keywords.apply(json.loads)
for index, i in zip(x_test.keywords.index, x_test.keywords):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    x_test.loc[index, 'keywords'] = str(list1)
for i, j in zip(x_test.keywords, x_test.index):
        list3 = []
        list3 = i
        x_test.loc[j, 'keywords'] = str(list3)
x_test.keywords = x_test.keywords.str.strip('[]').str.replace(' ', '').str.replace("'", '')
x_test.keywords = x_test.keywords.str.split(',')

for i in keywordsList.value_counts().keys().tolist()[:10] :
    x_test= derivesColumns(x_test,i,'keywords')

## changing the production_companies column from json to string
x_test.production_companies = x_test.production_companies.apply(json.loads)
for index, i in zip(x_test.production_companies.index, x_test.production_companies):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    x_test.loc[index, 'production_companies'] = str(list1)

for i, j in zip(x_test.production_companies, x_test.index):
        list4 = []
        list4 = i
        x_train.loc[j, 'production_companies'] = str(list4)
x_test.production_companies = x_test.production_companies.str.strip('[]').str.replace(' ', '').str.replace("'", '')
x_test.production_companies = x_test.production_companies.str.split(',')

for i in production_companiesList.value_counts().keys().tolist()[:10] :
    x_test= derivesColumns(x_test,i,'production_companies')

# changing the production_countries column from json to string
x_test.production_countries = x_test.production_countries.apply(json.loads)
for index, i in zip(x_test.production_countries.index,x_test.production_countries):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    x_test.loc[index, 'production_countries'] = str(list1)

for i, j in zip(x_test.production_countries, x_test.index):
        list5 = []
        list5 = i
        x_test.loc[j, 'production_countries'] = str(list5)
x_test.production_countries = x_test.production_countries.str.strip('[]').str.replace(' ', '').str.replace("'", '')
x_test.production_countries = x_test.production_countries.str.split(',')

for i in production_countriesList :
    x_test= derivesColumns(x_test,i,'production_countries')


# changing the spoken_languages column from json to string
x_test.spoken_languages = x_test.spoken_languages.apply(json.loads)
for index, i in zip(x_test.spoken_languages.index, x_test.spoken_languages):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    x_test.loc[index, 'spoken_languages'] = str(list1)

for i, j in zip(x_test.spoken_languages, x_test.index):
        list6 = []
        list6 = i
        x_test.loc[j, 'spoken_languages'] = str(list6)
x_test.spoken_languages = x_test.spoken_languages.str.strip('[]').str.replace(' ', '').str.replace("'", '')
x_test.spoken_languages = x_test.spoken_languages.str.split(',')

for i in spoken_languagesList :
    x_test= derivesColumns(x_test,i,'spoken_languages')

x_test['release_date'] = pd.to_datetime(x_test['release_date'].values).year

label_encoder = preprocessing.LabelEncoder()
labels_encoding = ['homepage', 'tagline', 'status', 'title', 'original_title', 'original_language']
for label in labels_encoding:
    x_test[label] = label_encoder.fit_transform(x_test[label])
x_test.runtime.replace(to_replace = 0, value = x_test.runtime.mean(), inplace=True)
x_test.budget.replace(to_replace = 0, value = x_test.budget.mean(), inplace=True)


columns=['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']
x_test =Feature_Encoder(x_test,columns)
#print(x_test.spoken_languages)


# preprocessing on revenue column
Z = x_test.loc[:, 'revenue']
scaler = MinMaxScaler(feature_range=(0, 100))
Z = Z.values.reshape(-1, 1)
standarized_revenue = scaler.fit_transform(Z)
standarized_revenue = scaler.fit_transform(Z)
x_test['revenue'] = standarized_revenue

# preprocessing on vote_count column
W = x_test.loc[:, 'vote_count']
scaler = MinMaxScaler(feature_range=(0, 100))
W = W.values.reshape(-1, 1)
standarized_vote_count = scaler.fit_transform(W)
x_test['vote_count'] = standarized_vote_count

# preprocessing on runtime column
F = x_test.loc[:, 'runtime']
scaler = MinMaxScaler(feature_range=(1, 100))
F = F.values.reshape(-1, 1)
standarized_runtime = scaler.fit_transform(F)
x_test['runtime'] = standarized_runtime
#print(standarized_runtime)
#print(data['runtime'])

# preprocessing on viewercount column
K = x_test.loc[:, 'viewercount']
scaler = MinMaxScaler(feature_range=(0, 100))
K = K.values.reshape(-1, 1)
standarized_viewer_count = scaler.fit_transform(K)
x_test['viewercount'] = standarized_viewer_count

# Define a grid of hyperparameters to search over
param_grid = {
    'text__vectorizer__max_features': [100, 500, 1000],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}

x_test = x_test[top_features.delete(-1)]

ypred= rf.predict(x_test)
print(' random forest ',accuracy_score(y_test, ypred))

ypred = reg.predict(x_test)
print(' logistic ',accuracy_score(y_test, ypred))

ypred = clf.predict(x_test)
print(' decision tree ',accuracy_score(y_test, ypred))

"""
def random_forest_modelGrid(x_train, x_test, y_train, y_test):
    # Number of trees in random forest
    rf = RandomForestClassifier()
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(x_train, y_train)
    prediction = rf_random.predict(x_test)
    true_value = np.asarray(y_test)[0]
    predicted_value = prediction[0]

    print('----------------------------------------------------------------------------'
          '----------------------------------------------------------------------------'
          '----------------------------------------------------------------------------')
    print(" Results from Grid Search ")
    print("\n The best estimator across ALL searched params:\n", rf_random.best_estimator_)
    print("\n The best score across ALL searched params:\n", rf_random.best_score_)
    print("\n The best parameters across ALL searched params:\n", rf_random.best_params_)
    print('Linear Regression Results:')
    # print('Co-efficient of multiple linear regression', sln.coef_)
    print('Mean Square Error to multiple linear regression', metrics.mean_squared_error(y_test, prediction))
    print('R2 score of test = ', r2_score(y_test, prediction))
    print('score of test = ', accuracy_score(y_test, prediction))
random_forest_modelGrid(x_train, x_test, y_train, y_test)
"""