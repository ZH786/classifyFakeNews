import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

rawdf_fke = pd.read_csv("/Users/zubairhussain/Desktop/Python/Datasets/Fake.csv")
rawdf_fke["Target"] = "FAKE"


rawdf_tre = pd.read_csv("/Users/zubairhussain/Desktop/Python/Datasets/True.csv")
rawdf_tre['Target'] = "TRUE"


rawdf= pd.concat([rawdf_fke,rawdf_tre])

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


