


import os 
import numpy as np 
import pandas as pd 
import string 
import string 
import re
import unicodedata 
import json 
from collections import Counter


import gc 
import tqdm 
from collections import defaultdict 
import scipy.sparse as sp 


# In[ ]:


# Update to point data_dir to folder contain train_set and test_set
data_dir = "/home/[]" 

train_set = pd.read_csv(os.path.join(data_dir, 'ift6390_arxiv/train.csv')) 
test_set = pd.read_csv(os.path.join(data_dir, 'ift6390_arxiv/test.csv')) 

# process
def filter_printable(input): 
    printable = set(string.printable) 
    return(''.join(filter(lambda x: x in printable, input))) 

def process(df, col): 
    df[col] = df[col].apply(lambda x: x.lower()) 
    df[col] = df[col].apply(lambda x: re.sub('\n', ' ', x) )
    df[col] = df[col].apply(lambda x: re.sub('-', ' ', x) )
    df[col] = df[col].apply(lambda x: re.sub('\\(', ' ', x) )
    df[col] = df[col].apply(lambda x: re.sub('\\)', ' ', x) ) 
    # in many cases it's a number in side < > for reference or footnote
    df[col] = df[col].apply(lambda x: re.sub("<.*?>", " ", x) )
    # punctuation
    df[col] = df[col].apply(lambda x: re.sub(r'\W',' ',x) ) 
    df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    df[col] = df[col].apply(lambda x: filter_printable(x)) 
    # added removal of pure numbers. This was not applied previously, and was added later after submission ended
    df[col] = df[col].apply(lambda x: re.sub(r' \d+', ' ', x)) 
    return df


# In[ ]:


# 
train_processed = process(train_set, 'Abstract') 
test_processed = process(test_set, 'Abstract') 
"""
# Generate and export a copy of dictionary for exploration if needed
# dictionary of word frequency 
def word_freq_dict(train_processed): 
        tmp_words = train_processed['Abstract'].apply(lambda x: x.split(' '))
        words = np.concatenate(np.array(tmp_words))
        words = list(words)
        word_dict = Counter(words)
        return word_dict 


word_dict = word_freq_dict(train_processed) 
# save 
json.dump(word_dict, open('word_dict.json', 'w')) 
"""


# In[ ]:


# Bernoulli 
class BernoulliVectorizer: 
    def __init__(self): 
        self.word_dict = {}
        self.word_dict_query = {} 
        self.wordvecs = []
    def word_freq_dict(self, dict_data): 
        
        tmp_words = dict_data['Abstract'].apply(lambda x: x.split(' '))
        words = np.concatenate(np.array(tmp_words))
        words = list(words)
        self.word_dict = Counter(words)
        return self.word_dict 
    def word_freq_dict_query(self, low_freq, high_freq): 
        for wd, value in self.word_dict.items(): 
            if value >= low_freq: 
                if value <= high_freq: 
                    if len(wd) > 3: 
                        self.word_dict_query[wd] = value
    
    def transform(self, input_data): 
        # by document. 0/1, no freq
        # use part of the word dict
        self.wordvecs = np.zeros(shape=((len(input_data), len(self.word_dict_query))))
        i = 0
        for doc in input_data['Abstract']: 
            tokens = doc.split(' ') 
            for word_idx in range(len(self.word_dict_query)): 
                if list(self.word_dict_query.keys())[word_idx] in tokens: 
                    self.wordvecs[i, word_idx] = 1
                else: 
                    pass
            i += 1 
            if i % 100 == 0: 
                print(i)
        return self.wordvecs
    def fit_transform(self, dict_data, input_data, low_freq=5, high_freq=2000): 
        self.word_freq_dict(dict_data) 
        self.word_freq_dict_query(low_freq, high_freq) 
        return self.transform(input_data) 


# In[ ]:


# 
bv = BernoulliVectorizer() 


train_wordvecs = bv.fit_transform(train_processed, train_processed, low_freq=50, high_freq=2000)
test_wordvecs = bv.fit_transform(train_processed, test_processed, low_freq=50, high_freq=2000)


# In[ ]:


###########
class BernoulliNB: 
    def __init__(self, adj = 1): 
        self.adj = adj 
    
    def fit(self, X, y): 
        self.n_classes = len(np.unique(y)) 
        
        self.classes = Counter(list(y)).keys() 
        self.counts = Counter(list(y)).values() 
        self.counts = np.array(list(self.counts)) / len(y) 
        
        # summarize data by class
        self.word_prob = np.zeros((self.n_classes, X.shape[1])) 
        for class_idx in range(len(self.classes)): 
                class_name = list(self.classes)[class_idx]
                x_idx = y[y == class_name].index
                for idx in x_idx: 
                    self.word_prob[class_idx] += X[idx] 
            
                
        self.word_prob += self.adj 
        class_sums = self.word_prob.sum(axis=1) + self.adj * self.n_classes
        self.word_prob = self.word_prob / class_sums[:, np.newaxis]
    
    def predict(self, X): 
         
        P = np.dot(X, np.log(self.word_prob).T) 
        P += np.log(self.counts) 
        P_max = np.argmax(P, axis=1) 
        P_max_class = [list(self.classes)[x] for x in P_max]
        return P_max_class


# In[ ]:


# Apply 
bnb = BernoulliNB(adj=1) 
bnb.fit(train_wordvecs, train_processed['Category'])
train_tmp_results = bnb.predict(train_wordvecs)

np.mean(train_tmp_results == train_processed['Category']) 

# predict test data
test_predict_results = bnb.predict(test_wordvecs)
output_prediction = pd.DataFrame(test_predict_results).reset_index().rename(
    columns={'index':'Id', 0:'Category'}) 

# Please update file path if output needed
# This exported list is uploaded to kaggle for submission 
output_prediction.to_csv('/home/[]/test_prediction_5.csv', index = False)







# In[ ]:


# Let's look at what's not matched in training data 
tmp_lst = train_processed['Category'][train_processed['Category'] != train_tmp_results]
Counter(tmp_lst)
"""
Counter({'astro-ph': 220,
         'hep-th': 50,
         'astro-ph.SR': 46,
         'gr-qc': 69,
         'stat.ML': 58,
         'astro-ph.CO': 171,
         'astro-ph.GA': 79,
         'cs.LG': 46,
         'physics.optics': 25,
         'quant-ph': 66,
         'cond-mat.mes-hall': 75,
         'hep-ph': 47,
         'cond-mat.mtrl-sci': 73,
         'math.AP': 10,
         'math.CO': 8})
"""


# In[ ]:


import pandas as pd 


# extract dictionary of word frequency 
def word_freq_dict(train_processed): 
        tmp_words = train_processed['Abstract'].apply(lambda x: x.split(' '))
        words = np.concatenate(np.array(tmp_words))
        words = list(words)
        word_dict = Counter(words)
        return word_dict 
# all categories in training
word_dict = word_freq_dict(train_processed) 

# 
all_words_df = pd.DataFrame(list(zip(word_dict.keys(), word_dict.values())), 
            columns = ['word', 'all_cat']) 
all_words_df = all_words_df.assign(length = all_words_df['word'].str.len()) 


all_cats = pd.unique(train_processed.loc[:, 'Category']) 
for cat in all_cats: 
    cat_dict = word_freq_dict(
    train_processed.query(
    ' Category == @cat '
    )
    )
    cat_word_df = pd.DataFrame(list(zip(cat_dict.keys(), cat_dict.values())), 
            columns = ['word', cat]) 
    all_words_df = pd.merge(all_words_df, cat_word_df, 
                      how='left', on = 'word') 
    
# A word that shows up in one category > 10 times, is defined as detected in the category. 
# if a word shows up in a category <= 10, it is treated as not detected in the cat. 
all_words_df = all_words_df.assign(
    n_cat = np.sum(all_words_df.loc[:, ['astro-ph', 'hep-ph', 'cs.LG', 'math.CO',
       'cond-mat.mes-hall', 'hep-th', 'stat.ML', 'physics.optics',
       'astro-ph.CO', 'gr-qc', 'astro-ph.SR', 'math.AP', 'cond-mat.mtrl-sci',
       'quant-ph', 'astro-ph.GA']] > 10, axis=1)
    )

all_words_summary = all_words_df.assign(
    # Compute whether a word shows up in one category > 50% of chances by counts
    # This will be used as a criteria to include low-frequency words. 
        over_half_in_cat = np.any(np.divide(all_words_df.loc[:, ['astro-ph', 'hep-ph', 'cs.LG', 'math.CO',
       'cond-mat.mes-hall', 'hep-th', 'stat.ML', 'physics.optics',
       'astro-ph.CO', 'gr-qc', 'astro-ph.SR', 'math.AP', 'cond-mat.mtrl-sci',
       'quant-ph', 'astro-ph.GA']], all_words_df.loc[:, 'all_cat'][:, np.newaxis] )
 > 0.5, axis=1)
        ).query(
    # total counts of a word in all categories
            'all_cat > 5'
            ).query(
    # words shorter or equal to 2 characters are not very meaningful/useful in predicting
    # While it was tempting to use 3 here, some words like "gas" did distinguish some categories well from others. 
            'length > 2'
            ).query(
    # If a word has total counts <= 10, it has to be in one category predominantly (> 50%)
                'n_cat > 0 | over_half_in_cat == True'
                ).query(
    # no features found in all categories
                    'n_cat < 15'
                    )

    




# In[ ]:


# generate a word dict in format required by "BernoulliVectorizer"
hand_crafted_word_dict = dict(
    zip(all_words_summary.loc[:, 'word'], all_words_summary.loc[:, 'all_cat'])
    )

bv_2 = BernoulliVectorizer() 

bv_2.word_dict_query = hand_crafted_word_dict

# split train into train and validation 
# 

train_wordvecs = bv_2.transform(train_processed.iloc[:6000, :]) # since paper categories are already randomized
val_wordvecs = bv_2.transform(train_processed.iloc[6000:, :])

bnb = BernoulliNB(adj=1) 
bnb.fit(train_wordvecs, train_processed.loc[:5999, 'Category'])
val_tmp_results = bnb.predict(val_wordvecs)

np.mean(val_tmp_results == train_processed.loc[6000:, 'Category']) 
# 0.7893333333333333 before removing pure number through pre-processing 
# 0.7906666666666666 
# This is a good improvement without using low frequency words, leading to overfitting. 


# In[ ]:


# Other models
# example code

import numpy as np 
import pandas as pd 
import os 
import re 
import nltk 
from sklearn.datasets import load_files 
nltk.download('stopwords') 
import pickle 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import string

### test multinomial Naive Bayes 
from sklearn.feature_extraction.text import CountVectorizer 

vectorizer = CountVectorizer(max_features=None, min_df = 5, 
                                   max_df = 0.8, 
                                   ngram_range=(1,3), 
                                   strip_accents='unicode', 
                                   stop_words=stopwords.words('english'))


train_dict = vectorizer.fit(train_processed['Abstract'], y = train_processed['Category'])

train_vecs = vectorizer.transform(train_processed['Abstract']).toarray() 
test_vecs = vectorizer.transform(test_processed['Abstract']).toarray()

from sklearn.feature_extraction.text import TfidfTransformer 
tfidfconverter = TfidfTransformer() 

train_vecs = tfidfconverter.fit_transform(train_vecs).toarray()
test_vecs = tfidfconverter.fit_transform(test_vecs).toarray()
#
from sklearn.naive_bayes import MultinomialNB 

MNNB_classifier = MultinomialNB(alpha=0.95, fit_prior=False) 

MNNB_classifier.fit(train_vecs, train_processed['Category']) 

test_pred = MNNB_classifier.predict(test_vecs)
# export 
output_prediction = pd.DataFrame(test_pred).reset_index().rename(
    columns={'index':'Id', 0:'Category'})
# Update file path if needed
output_prediction.to_csv('/home/jx/Documents/IFT_6390/competition_1/test_prediction_8.csv', index = False)


