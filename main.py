import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics, linear_model, __all__, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
import datetime
import json
from xgboost import XGBClassifier
from sklearn.svm import SVR

data = pd.read_csv('movies-regression-dataset.csv')

X = data.iloc[:, 0:19]
Y = data['vote_average']

# split the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)

#print("duplicates of id: ",x_train.id.duplicated().sum(),'\n')
#print("nulls of release_date: ",x_train.release_date.isnull().sum())
x_train['release_date'] = pd.to_datetime(x_train['release_date'].values).year

x_train['runtime'].fillna(value=x_train['runtime'].mean(), inplace=True)
x_train['budget'].fillna(value=x_train['budget'].mean(), inplace=True)
x_train['revenue'].fillna(value=x_train['revenue'].mean(), inplace=True)

x_train.runtime.replace(to_replace = 0, value = x_train.runtime.mean(), inplace=True)
x_train.budget.replace(to_replace = 0, value = x_train.budget.mean(), inplace=True)


for col in x_train.columns:
    x_train[col] = x_train[col].fillna(x_train[col].mode()[0])

label_encoder = preprocessing.LabelEncoder()
labels_encoding = ['homepage', 'tagline', 'status', 'title', 'original_title', 'original_language']

le = LabelEncoder()

encoders = {}

for col in labels_encoding:
    if x_train[col].dtype == 'object':
        unique_values = list(x_train[col].unique())
        unique_values.append('Unseen')
        le = LabelEncoder().fit(unique_values)
        x_train[col] = le.transform(x_train[[col]])
        encoders[col] = le


# changing the genres column from json to string
x_train.genres = x_train.genres.apply(json.loads) #json loads is used for converting the json format to python object
for index, i in zip(x_train.genres.index, x_train.genres):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    x_train.loc[index, 'genres'] = str(list1)

# changing the string to list format to get all the genres in each cell individually
for i, j in zip(x_train.genres, x_train.index):
        list2 = []
        list2 = i
        x_train.loc[j, 'genres'] = str(list2)
x_train.genres = x_train.genres.str.strip('[]').str.replace(' ', '').str.replace("'", '')
x_train.genres = x_train.genres.str.split(',')

#to get the unique values of genres
genreList = []
for index, row in x_train.iterrows(): #iterrow let us iterate on each row in the dataframe
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
#print(x_train.spoken_languages)


# preprocessing on revenue column
Z = x_train.loc[:, 'revenue']
revenueScaler=MinMaxScaler(feature_range=(0, 100))
Z = Z.values.reshape(-1, 1)
standarized_revenue = revenueScaler.fit_transform(Z)
x_train['revenue'] = standarized_revenue

# preprocessing on vote_count column
W = x_train.loc[:, 'vote_count']
vote_countscaler = MinMaxScaler(feature_range=(0, 100))
W = W.values.reshape(-1, 1)
standarized_vote_count = vote_countscaler.fit_transform(W)
x_train['vote_count'] = standarized_vote_count

# preprocessing on runtime column
F = x_train.loc[:, 'runtime']
runtimescaler = MinMaxScaler(feature_range=(1, 100))
F = F.values.reshape(-1, 1)
standarized_runtime = runtimescaler.fit_transform(F)
x_train['runtime'] = standarized_runtime


# preprocessing on viewercount column
K = x_train.loc[:, 'viewercount']
viewercountscaler = MinMaxScaler(feature_range=(0, 100))
K = K.values.reshape(-1, 1)
standarized_viewer_count = viewercountscaler.fit_transform(K)
x_train['viewercount'] = standarized_viewer_count


data = x_train
data['vote_average'] = y_train
corr = data.corr()
top_features = corr.index[abs(corr['vote_average']) > 0.1]
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_features.delete(-1)
x_train = data[top_features.delete(-1)]

#################################################modeling

x_train.replace([np.inf, -np.inf], np.nan, inplace=True)

mlr = LinearRegression()
x_train = x_train[top_features.delete(-1)]
mlr.fit(x_train, y_train)


clf = linear_model.Lasso(alpha=0.1)
clf.fit(x_train,y_train)


regressor1 = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=5,max_leaf_nodes = 1000)
regressor1.fit(x_train, y_train)



######################################################################test
# changing the genres column from json to string
# datatest = pd.read_csv('movies-regression-dataset.csv')
#
# x_test = datatest.iloc[:, 0:19]
# y_test = datatest['vote_average']
x_test['release_date'] = pd.to_datetime(x_test['release_date'].values).year

x_test['runtime'].fillna(value=x_test['runtime'].mean(), inplace=True)
x_test['budget'].fillna(value=x_test['budget'].mean(), inplace=True)
x_test['revenue'].fillna(value=x_test['revenue'].mean(), inplace=True)

x_test.runtime.replace(to_replace = 0, value = x_test.runtime.mean(), inplace=True)
x_test.budget.replace(to_replace = 0, value = x_test.budget.mean(), inplace=True)


for col in x_test.columns:
    x_test[col] = x_test[col].fillna(x_test[col].mode()[0])

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

for col in labels_encoding:
    if x_test[col].dtype == 'object':
        le = encoders.get(col)
        x_test[col] = [x if x in le.classes_ else 'Unseen' for x in x_test[col]]
        x_test[col] = le.transform(x_test[[col]])



columns=['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']
x_test =Feature_Encoder(x_test,columns)
#print(x_test.spoken_languages)


# preprocessing on revenue column
Z = x_test.loc[:, 'revenue']
Z = Z.values.reshape(-1, 1)
standarized_revenue = revenueScaler.transform(Z)
x_test['revenue'] = standarized_revenue

# preprocessing on vote_count column
W = x_test.loc[:, 'vote_count']
W = W.values.reshape(-1, 1)
standarized_vote_count = vote_countscaler.transform(W)
x_test['vote_count'] = standarized_vote_count

# preprocessing on runtime column
F = x_test.loc[:, 'runtime']
F = F.values.reshape(-1, 1)
standarized_runtime = runtimescaler.transform(F)
x_test['runtime'] = standarized_runtime
#print(standarized_runtime)
#print(data['runtime'])

# preprocessing on viewercount column
K = x_test.loc[:, 'viewercount']
K = K.values.reshape(-1, 1)
standarized_viewer_count = viewercountscaler.transform(K)
x_test['viewercount'] = standarized_viewer_count

############################################modeling

x_test = x_test[top_features.delete(-1)]

ypred= mlr.predict(x_test)
mse = mean_squared_error(y_test, ypred)
r2= r2_score(y_test,ypred)
print('Mean Square Error of mlr : {:.4f}'.format(mse))
print('r2 score for perfect mlr is : {:.4f}'.format(r2),'\n')
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x=y_test, y=ypred, color= 'red')
plt.xlabel('True Values for mlr', fontsize=15)
plt.ylabel('Predictions for mlr', fontsize=15)
plt.ylim(0,)
plt.show()


ypred= clf.predict(x_test)
mse = mean_squared_error(y_test, ypred)
r2= r2_score(y_test,ypred)
print('Mean Square Error of lasso : {:.4f}'.format(mse))
print('r2 score for perfect lasso is : {:.4f}'.format(r2),'\n')
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x=y_test, y=ypred,color= 'red')
plt.xlabel('True Values for lasso', fontsize=15)
plt.ylabel('Predictions for lasso', fontsize=15)
plt.ylim(0,)
plt.show()



ypred= regressor1.predict(x_test)
r2= r2_score(y_test,ypred)
mse = mean_squared_error(y_test, ypred)
print('Mean Square Error of random : {:.4f}'.format(mse))
print('r2 score for perfect random forest is : {:.4f}'.format(r2),'\n')
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x=y_test, y=ypred,color= 'red')
plt.xlabel('True Values for random', fontsize=15)
plt.ylabel('Predictions for random', fontsize=15)
plt.ylim(0,)
plt.show()








