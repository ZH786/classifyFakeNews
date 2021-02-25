import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize, TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords
import nltk
import spacy
import en_core_web_sm

rawdf_fke = pd.read_csv("/Users/zubairhussain/Desktop/Python/Datasets/Fake.csv")
rawdf_fke["Target"] = "FAKE"


rawdf_tre = pd.read_csv("/Users/zubairhussain/Desktop/Python/Datasets/True.csv")
rawdf_tre['Target'] = "TRUE"


rawdf= pd.concat([rawdf_fke,rawdf_tre]) #equivalent to rbind() 

#shuffle dataframe
rawdf=rawdf.sample(frac=1)

rawdf.info()
rawdf.head()
rawdf.describe()
type(rawdf)

rawdf.Target.value_counts() #fairly balanced dataset
rawdf.isnull().sum() # no missing values

#no empty whitespace texts in data
empty=[]
for i,j,k in rawdf[['title','text']].itertuples():
    if j.isspace():
        empty.append(i)

len(empty)

#Count text length
rawdf['textLength']=rawdf.text.str.split().apply(len)
rawdf.textLength.describe() #max 8135, avg 405 words, min 0????

#EDA
sns.histplot(data=rawdf, x = "textLength", binwidth=100)
plt.xlim(0,4000)
plt.show()

rawdf=rawdf[rawdf.textLength!=0][['text','Target','textLength']]


def depunctuate(text):

    punc=[]

    for i in text:
        if i not in string.punctuation:
            punc.append(i)
    clean_punc = ''.join(punc)

    return(clean_punc)
    
rawdf['text']=rawdf.text.apply(depunctuate)

#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
#nltk.download('punkt')

stop_words = stopwords.words('english')

newstopwords = [depunctuate(word) for word in stop_words]

tknzr = TreebankWordTokenizer()
rawdf['text']=rawdf['text'].apply(tknzr.tokenize)

def rmStopWords(text):
    clean = []
    for i in text:
        if i not in stop_words:
            clean.append(i)
    cleaned = ' '.join(clean)        
    return(cleaned)        

rawdf['text']=rawdf['text'].apply(rmStopWords) # stop words removed from text field

#Modelling
X = rawdf.text
y = rawdf.Target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#create documentTerm Matrix and TFIDF
#vect = TfidfVectorizer()
#X_train_tfidf = vect.fit_transform(X_train)

#SKLEARN pipeline
txt_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])

txt_clf.fit(X_train, y_train)
clf_pred = txt_clf.predict(X_test)

print(confusion_matrix(y_test, clf_pred))
print(classification_report(y_test,clf_pred))