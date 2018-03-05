'''
FWAF - Machine Learning driven Web Application Firewall
Author: Faizan Ahmad
Performance improvements: Timo Mechsner
Website: http://fsecurify.com
'''

from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import urllib.parse
import pandas as pd

#import matplotlib.pyplot as plt

def loadFile(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    with open(filepath,'r') as f:
        data = f.readlines()
    data = list(set(data))
    result = []
    for d in data:
        d = str(urllib.parse.unquote(d))   #converting url encoded data to simple string
        result.append(d)
    return result

badQueries = loadFile('malicious.txt')
validQueries = loadFile('normal.txt')

badQueries = list(set(badQueries))
validQueries = list(set(validQueries))
allQueries = badQueries + validQueries
yBad = [1 for i in range(0, len(badQueries))]  #labels, 1 for malicious and 0 for clean
yGood = [0 for i in range(0, len(validQueries))]
y = yBad + yGood
queries = allQueries

vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3)) #converting data to vectors
X = vectorizer.fit_transform(queries)


#######

#Build a logitical regression model using TFIDF over n-grams


#different from task 2, we use TFIDF values of 1-gram, 2-gram and 3-gram tokens
vectorizer_new = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3))
tfidf_matrix = vectorizer_new.fit_transform(queries)
feature_names = vectorizer_new.get_feature_names()

dense = tfidf_matrix.todense()
denselist = dense.tolist()
df_new = pd.DataFrame(denselist, columns=feature_names)


msg_train, msg_test, label_train, label_test=train_test_split(df_new, y, test_size=0.2)


print (len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

model = LogisticRegression().fit(msg_train, label_train)
#model3.predict(msg_test)

# Predict the classes of the testing data set
class_predict = model.predict(msg_test)

# Compare the predicted classes to the actual test classes
print ("customized auc result: %d" %metrics.accuracy_score(label_test,class_predict))
#######


