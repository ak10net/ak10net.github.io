---
layout: post
title: "Content based song recommender"
author: "Ankit"
tags: recommender content-based
excerpt_separator: <!--more-->
---

## We will try to recommend songs with similar lyrics<!--more-->

`If you love music this post is for you`

## What is content based filtering?
Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback. 
Here we will create item features from song lyrics and will try to recommned similar song to what user liked.

+ Advantages of content based recommneder
	+ The model doesn't need any data about other users, since the recommendations are specific to this user. This makes it easier to scale to a large number of users.

	+ The model can capture the specific interests of a user, and can recommend niche items that very few other users are interested in.

+ Disadvantages of content based recommender
	+ Since the feature representation of the items are hand-engineered to some extent, this technique requires a lot of domain knowledge. Therefore, the model can only be as good as the hand-engineered features.

	+ The model can only make recommendations based on existing interests of the user. In other words, the model has limited ability to expand on the users' existing interests.

### About the dataset
+ Dataset we are going to use is from [Genius.com](https://www.cs.cornell.edu/~arb/data/genius-expertise/)
+ We will use `lyrics.jl` file from the download
+ Dataset has two columns 'song' and 'lyrics'
+ Song column contains artist name plus name of the song
+ Lyrics column contains lyrics as displayed on genius.com
+ We are going to clean the data set and extract new features before building recommender

![Dataset](/assets/content_recommender2.png)

Let's start with basic imports
```python
from collections import Counter
import pandas as pd
import re
from langdetect import detect
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter
%matplotlib inline
pd.set_option('max_colwidth', 400)
```

Loading data and filtering
```python
df = pd.read_json('lyrics.jl', lines=True)
```
We create new columns for artist from song column
Also, we are going to use on songs from english language so we remove songs of other language after language detection
A good sanity check is to know song lengths, while doing that we found out that some songs have length upto 30k and some as low as 5 words. We removed songs which had song length below and above cutoffs

```python
def lang_detector(x):
    '''
        takes a string and returns language of string
    '''
    try:
        return detect(x)
    except:
        return 'unknown language'

def cleaner(df):
    '''
        cleans song column
        extract artist from song column
        cleans lyrics column
        applies lang_detector function 
        creates new column for song length
        removes songs of laguage other than english
        removes outliers calculated using song length
        creates new column of song_length_quantiles for viz
        returns cleand df
    '''
    df['artist'] = df['song'].apply(lambda x: ' '.join(x.split('-')[:2]))
    df['song'] = df['song'].apply(lambda x: x[:-7])    
    df['lyrics'] = df['lyrics'].apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x))
    df['lyrics'] = df['lyrics'].apply(lambda x: re.sub('\n\n', '', x))
    df['lyrics'] = df['lyrics'].apply(lambda x: re.sub('\n', ' ', x))
    df['lang'] = df['lyrics'].apply(lambda x: lang_detector(x))
    df = df[df['lang'] == 'en']
    df['song_length'] = df['lyrics'].apply(lambda x: len(x))
    Q1 = df['song_length'].quantile(0.25)
    Q3 = df['song_length'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['song_length'] > (Q1 - 1.5 * IQR)) & (df['song_length'] < (Q3 + 1.5 * IQR))]
    df['song_length_quantiles'] = pd.cut(df['song_length'], bins=10, precision=0)    
    return df

df = cleaner(df)
```

We have created song length quantiles from song length column. Now we can visualize the distribution of song length?
```python
song_len_df = df['song_length_quantiles'].value_counts(sort=True).reset_index().rename(columns={'index': 'quantiles', 'song_length_quantiles':'count'})
sns.barplot(x='quantiles', y='count', data=song_len_df,   palette="Blues_d")
plt.xticks(rotation=45)
plt.show()
```
![Song length distribution](/assets/content_recomender.png)

In EDA we found out that datasets contains lot of songs of few artists and very few of some artist. At the same time dataset contains songs from 12K unique artists. We only want songs recommended from different artist. Thus dropping songs from same artists.
```python
df = df.drop_duplicates(subset='artist', keep='first')
df = df.reset_index(level=0)
df['id'] = df['index']
df = df.drop(['index'], axis=1)
```

We will convert lyrics to [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) matrix before calculating cosine similarity between songs and storing similar 50 songs in results dict. 
```python
# Calculating cosine similarities from lyrics and storing similar song results in results dict
tf = TfidfVectorizer(analyzer='word', min_df=0, max_features= 100 ,stop_words='english', lowercase=True)
tfidf_matrix = tf.fit_transform(df['lyrics'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
results = {}

for idx, row in df.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-50:-1]
    similar_items = [(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices]
    results[row['id']] = similar_items[1:]
```

Now that we have recommendations for every song in the dataset. We can query the dict for similar songs as per user request
```python
def item(id):
    return df.loc[df['id'] == id]['song']

def recommend(id, num):
    print("Recommending " + str(num) + " songs similar to " + item(id))
    print("\n")
    recs = results[id][:num]
    i=0
    for rec in recs:
        print("We recommend : " + item(rec[1]) + " (score:" + str(rec[0]) + ")")

recommend(14,3)
```
Final output
```
2    Recommending 3 songs similar to Travis-scott-goosebumps
Name: song, dtype: object
6954    We recommend : Tyga-u-cry (score:0.9022309682343544)
Name: song, dtype: object
1819    We recommend : Goldlink-u-say (score:0.8874335491812315)
Name: song, dtype: object
1662    We recommend : Logic-time-machine (score:0.8612878563854101)
Name: song, dtype: object
```

Seems like recommendations are bang on with high confidence. Model is recommending artist in similar
genre and those that have same style of music.

I love listening to new music and love spotify's recommendation. With this exercise, I wanted to see if simple model like this can be good at song suggestion. Apparently, simple models can also work.
