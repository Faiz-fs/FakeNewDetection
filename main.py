import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

sns.set()

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
y_test = pd.read_csv('data/submit.csv')

test['label'] = y_test['label']

# print(train.head())

# print(train.info())

train.dropna(axis=0, how='any', inplace=True)
test.dropna(axis=0, how='any', inplace=True)

# print('Number of NaN values in sets: ', train.isna().sum().sum() + test.isna().sum().sum())

map_values = {
    0: 'Real',
    1: 'Fake'
}

df_t = train.label.value_counts()
df_t.index = df_t.index.map(map_values)

# plt.figure(figsize=(12, 6))
# df_t.plot(kind='bar', fontsize=14)
# plt.title("Real vs fake news", fontsize=20)
# plt.show()

vectorizer = TfidfVectorizer()

train_vecotorizer = vectorizer.fit_transform(train['text'].astype('U'))
test_vecotorizer = vectorizer.transform(test['text'].astype('U'))

pac = PassiveAggressiveClassifier(max_iter=88)
pac.fit(train_vecotorizer, train['label'])

pac_score = pac.score(test_vecotorizer, test['label'].values)
# print(pac_score)

log_reg = LogisticRegression()

log_reg.fit(train_vecotorizer, train['label'])
lg_score = log_reg.score(test_vecotorizer, test['label'])
# print(lg_score)

rando = RandomForestClassifier(n_estimators=5)
rando.fit(train_vecotorizer, train['label'])
rando_score = rando.score(test_vecotorizer, test['label'])
# print(rando_score)

classifier = MultinomialNB()
classifier.fit(train_vecotorizer, train['label'])
multi_score = classifier.score(test_vecotorizer, test['label'])
# print(multi_score)

models_value = {'PassiveAggressiveClassifier': pac_score, 'Logistic Regression': lg_score,
                'RandomForestClassifier': rando_score, 'MultinomialNB': multi_score}
models = pd.Series(models_value).sort_values(ascending=False)
# print(models)

models_list = [log_reg, rando, pac, classifier]
'''
plt.figure(figsize=(6, 4))
models.plot(kind='bar')
plt.title('Accuracy of models')
plt.show()'''

y_pred_rando = rando.predict(test_vecotorizer)
y_pred_pca = pac.predict(test_vecotorizer)

cm_pac = confusion_matrix(y_pred_rando, test['label'])
cm_rando = confusion_matrix(y_pred_pca, test['label'])

# plot_confusion_matrix(cm_pac)
# plt.show()

# plot_confusion_matrix(cm_rando)
# plt.show()

fake = pd.read_csv('data/production/Fake.csv')
true = pd.read_csv('data/production/True.csv')


# print(true.head())

def predict_data(data, vectorizer, classifier):
    text = vectorizer.transform(data.astype('U'))
    y_pred = classifier.predict(text)
    return (round(y_pred.sum() / len(data) * 100, 2))


def exec_models(models_name, models):
    df = pd.DataFrame(data=0, columns=['Fake news predict as fake', 'True news predict as true'], index=models_name)

    for i in range(len(models)):
        f = predict_data(fake['text'], vectorizer, models[i])
        t = predict_data(true['text'], vectorizer, models[i])

        df.loc[models_name[i]] = [f, t]

        print('\n{}'.format(models_name[i]))
        print("Fake news predict as fake: {}%".format(f))
        print("True news predict as true: {}%".format(100 - t))

    return df


df_result = exec_models(list(models.index), models_list)
# print(df_result)

df_result['mean'] = df_result[['Fake news predict as fake', 'True news predict as true']].mean(axis=1)
df_result = df_result.sort_values(by='mean', ascending=False)

plt.figure(figsize=(6, 4))
_ = df_result['mean'].plot(kind='bar')
plt.show()

pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(pac, open('model.pkl', 'wb'))

vec = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
msg1 = pd.DataFrame(index=[0], data=true['text'][100], columns=['data'])
text = vec.transform(msg1['data'].astype('U'))
result = model.predict(text)

print("Real (0) Fake (1) news : {}".format( result[0]  ))

# print("Real (0) Fake (1) news : {}".format(result[0]  ))
msg = pd.DataFrame(index=[0], data=fake['text'][100], columns=['data'])
text = vec.transform(msg['data'].astype('U'))
result = model.predict(text)

print("Real (0) Fake (1) news : {}".format(result[0]))
