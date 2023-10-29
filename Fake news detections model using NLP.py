#!/usr/bin/env python
# coding: utf-8

# In[1]:


### import necessary stuffs


# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv(r"/Users/nikhildonde/Downloads/train.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


## drop your missing values
df.dropna(inplace=True)


# In[8]:


df.shape


# In[ ]:





# In[9]:


## checking distribution of data
import seaborn as sns
def create_distribution(feature):
    return sns.countplot(df[feature])


# In[10]:


df.dtypes


# In[11]:


df['label']=df['label'].astype(str)


# In[12]:


df.dtypes


# In[13]:


create_distribution('label')


# In[ ]:





# In[14]:


df.head(20)


# In[15]:


messages=df.copy()


# In[16]:


#why to rset_index,bcz in above we can check,when we drop our rows get deleted as 6 and 8th so to make it in a order , we have to use reset_index

messages.reset_index(inplace=True)


# In[17]:


messages.head(10)


# In[18]:


messages.drop(['index','id'],axis=1,inplace=True)


# In[19]:


messages.head()


# In[20]:


#note we will consider only title for pre-processing


# In[21]:


data=messages['title'][0]
data


# In[ ]:





# In[22]:


import re


# In[23]:


re.sub('[^a-zA-Z]',' ', data)


# In[24]:


data=data.lower()
data


# In[25]:


list=data.split()
list


# In[26]:


get_ipython().system('pip install nltk')


# In[27]:


import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[28]:


ps=PorterStemmer()


# In[29]:


review=[ps.stem(word) for word in list if word not in set(stopwords.words('english'))]
review


# In[30]:


review=[]
for word in list:
    if word not in set(stopwords.words('english')):
        review.append(ps.stem(word))
review


# In[31]:


' '.join(review)


# In[32]:


### lets do same task for each & every row


# In[33]:


corpus=[]
sentences=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ', messages['title'][i])
    review=review.lower()
    list=review.split()
    review=[ps.stem(word) for word in list if word not in set(stopwords.words('english'))]
    sentences=' '.join(review)
    corpus.append(sentences)


# In[ ]:





# In[34]:


corpus[0]


# In[35]:


corpus


# In[36]:


len(corpus)


# In[ ]:





# In[37]:


## Applying Countvectorizer
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer


# In[38]:


## max_features=5000, it means I just need top 5000 features 
#example ABC News is basically 2 words,so in ngram,i have Given (1,3),so it will take the combination of 1 word,then 2 words 
#then 3 words

cv=CountVectorizer(max_features=5000,ngram_range=(1,3))


# In[39]:


X=cv.fit_transform(corpus).toarray()


# In[40]:


X.shape
#ie we get 5000 features now


# In[41]:


X


# In[ ]:





# In[42]:


cv.get_feature_names_out()[0:20]


# In[43]:


messages.columns


# In[44]:


y=messages['label']


# In[ ]:





# In[45]:


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split


# In[46]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=42)


# In[47]:


X_test


# In[48]:


X_test.shape


# In[ ]:





# ###  MultinomialNB Algo

# In[49]:


#this algo works well with text data

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[50]:


classifier.fit(X_train,y_train)


# In[51]:


pred=classifier.predict(X_test)
pred


# In[ ]:





# In[52]:


from sklearn import metrics


# In[53]:


metrics.accuracy_score(y_test,pred)


# In[54]:


cm=metrics.confusion_matrix(y_test,pred)
cm


# In[55]:


import matplotlib.pyplot as plt
import numpy as np


# In[56]:


### make your confusion amtrix more user-friendly

plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Confusion Matrix')
labels=['positive','negative']
tick_marks=np.arange(len(labels))
plt.xticks(tick_marks,labels)
plt.yticks(tick_marks,labels)


# In[57]:


labels=['positive','negative']
np.arange(len(labels))


# In[58]:


def plot_confusion_matrix(cm):
    plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Confusion Matrix')
    labels=['positive','negative']
    tick_marks=np.arange(len(labels))
    plt.xticks(tick_marks,labels)
    plt.yticks(tick_marks,labels)


# In[59]:


plot_confusion_matrix(cm)


# In[ ]:





# ### Passive Aggressive Classifier Algorithm

# In[60]:


#this algo works well with text data and is basica0lly used for text data


# In[61]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[62]:


linear_clf=PassiveAggressiveClassifier()


# In[63]:


linear_clf.fit(X_train,y_train)


# In[64]:


predictions=linear_clf.predict(X_test)


# In[65]:


metrics.accuracy_score(y_test,predictions)


# In[66]:


cm2=metrics.confusion_matrix(y_test,predictions)
cm2


# In[67]:


plot_confusion_matrix(cm2)


# In[ ]:





# In[68]:


## Get Features names
#to detect which fake and which is most real word

feature_names=cv.get_feature_names_out()


# In[ ]:





# In[69]:


#most negative value is most fake word,if we go towards lower value in -ve,ie we have most fake value
classifier.feature_log_prob_[0]


# In[ ]:





# In[70]:


### Most 20 real values
sorted(zip(classifier.feature_log_prob_[0],feature_names),reverse=True)[0:20]


# In[ ]:





# In[ ]:




