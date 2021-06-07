---
layout: post
title: "Aspect based sentiment analysis"
author: "Ankit"
tags: sentiment spacy aspect
excerpt_separator: <!--more-->
---
### Sentiment analysis with a flavour of aspect.<!--more-->

We have been using sentiment of customer feedback and review to make future decisions. But, taking review sentiment as a whole can be misleading. A review can be about multiple aspect and in this post we are going to aspects of review and sentiment related to those aspects.
We are going to use [Amazon food reviews dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews)

### What is aspect based sentiment analysis?

Aspect-based sentiment analysis (ABSA) is a text analysis technique that categorizes data by aspect and identifies the sentiment attributed to each one. Aspect-based sentiment analysis can be used to analyze customer feedback by associating specific sentiments with different aspects of a product or service.

Sentiments: positive or negative opinions about a particular aspect
Aspects: the category, feature, or topic that is being talked about

### Why aspect based sentiment analysis?

let’s assume you’re trying to classify a single yelp restaurant review into one of five aspects: food, service, price, ambience, or simply anecdotal/miscellaneous. You could label the entire review and say that it mentions both food and price. At this level, you would have the most information about the context to make an accurate prediction, but it may require extra steps if you wanted to find out which particular sentence or word is referring to the specific aspect. Also, a document that mostly talks about food with one brief mention of price would be categorized in the same group as one that mostly talks about price and very little of food. At the individual word level, you have the most specificity; maybe a person was dissatisfied particularly with the music, which would be a feature of ambience. However, you may lose the context around the word to really parse out the more accurate context in deriving the sentiment around the aspect.

Basic imports 
```python
# Basic imports
import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
import spacy
nlp = spacy.load("en_core_web_sm")
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from collections import Counter, defaultdict
from contractions_dict import contraction_mapping
import altair as alt
from textblob import TextBlob
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
```

Out of 500K reviews we are going to use 50k reviews for analysis. Also, we will take equal amount of postive and negative reqviews for analysis.
```python
# Out of full dataset we are going to take 10% of reviews ie 50000 for analysis
df=pd.read_csv("Reviews.csv",nrows=200000)

# Dropping columns that aren't useful from nlp perspective
df = df.drop(columns=['Id','ProductId', 'UserId', 'ProfileName', 'Time', 'HelpfulnessNumerator', 'HelpfulnessDenominator'], axis=1)

# Dropping duplicates and na 
df.drop_duplicates(subset=['Text'],inplace=True)
df.dropna(axis=0,inplace=True)

# Reviews with score 4 and 5 are of positive sentiment and 1,2,3 score are of negative sentiment  
df['Sentiment'] = df['Score'].apply(lambda x: 1 if x in [4,5] else 0 )
pos = df[df['Sentiment']==1][:25000]
neg = df[df['Sentiment']==0][:25000]
df = pd.concat([pos,neg])
# We don't need Score columns anymore
df = df.drop(columns = ['Score'], axis=1)
```

We will clean up the raw reviews with basic NLP pipeline
```python
stop_words = set(stopwords.words('english')) 

def text_cleaner(text,num):
    '''
        Text cleaner does the following
        1. Lowercase text
        2. Removes non text from raw reviews
        3. Substitutes not alphanumeric characters
        4. Correct words using the contractions mapping dictionary
        5. Removes Junk characters generated after cleaning
        6. Remove stop words if num=0 that means for review only not for summary
        
        Parameters: String, Number
        Returns: String
    '''
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num==0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens=newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:                                                
            long_words.append(i)   
    return (" ".join(long_words)).strip()

# Cleaning raw reviews
cleaned_text = []
for t in df['Text']:
    cleaned_text.append(text_cleaner(t,1)) 

# Cleaning review summaries
cleaned_summary = []
for t in df['Summary']:
    cleaned_summary.append(text_cleaner(t,1))

# Create news columns for cleaned data  
df['cleaned_text']=cleaned_text
df['cleaned_summary']=cleaned_summary

# Dropping empty rows
df.replace('', np.nan, inplace=True)
df.dropna(axis=0,inplace=True)

# Dropping raw summary and review columns 
df = df.drop(columns=['Summary', 'Text'], axis=1)
```
Our dataset has sentiment and cleaned reviews.

![png](/aasets/amazon_reviews.png)

Now we are going to extract aspect of the review using dependency parser and also we will extract words define sentiment for that aspect. With these descriptors we will predict sentiment of that aspect. Note sentiment of overall sentence and aspect may differ. We will see how much is the difference by looking at confusion matrix later on.
```python
def aspect_description_extractor(sentence):
    '''
        Aspect description extractors
        1. Parses the sentence with spacy parser
        2. If token is POS is noun and dependence is noun subject
        3. Then it considers that token as aspect of review
        4. If POS of token is adjective and child token POS is adverb 
        5. Then added children token and tokes as descrition
        
        Parameters: String
        Returns: Dict
    '''
    result = []
    doc = nlp(sentence)
    descriptive_term = ''
    target = ''
    for token in doc:
        if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
            target = token.text
        if token.pos_ == 'ADJ':
            prepend = ''
            for child in token.children:
                if child.pos_ != 'ADV':
                    continue
                prepend += child.text + ' '
            descriptive_term = prepend + token.text
    result.append({'aspect': target,'description': descriptive_term})
    return result

# Applying extractor to cleaned text
df['result'] = df['cleaned_text'].apply(lambda x: aspect_description_extractor(x))

# Creating new columns for aspect and description
df['aspect'] = df['result'].apply(lambda x: x[0]['aspect'])
df['description'] = df['result'].apply(lambda x: x[0]['description'])
df['predicted_sentiment'] = df['description'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity>0 else 0)
```

In Amazon food reviews what are the most reviewed aspects and how is there sentiment
```python
top_aspects = df['aspect'].value_counts().head(10).index
temp1 = df[df['aspect'].isin(top_aspects)].groupby(['aspect', 'Sentiment']).size().reset_index(name='count')
alt.Chart(temp1, title='Aspect extracted from product reviews').mark_bar().encode(
    alt.X('Sentiment:O'),
    alt.Y("count"),
    color='Sentiment:N',
    column='aspect'
).properties(width=60)
```





<div id="altair-viz-46d59fbe6a3e481b96ef21382c6b4693"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-46d59fbe6a3e481b96ef21382c6b4693") {
      outputDiv = document.getElementById("altair-viz-46d59fbe6a3e481b96ef21382c6b4693");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-16f6c0c0f8c4d4dad5efcc988033684b"}, "mark": "bar", "encoding": {"color": {"type": "nominal", "field": "Sentiment"}, "column": {"type": "nominal", "field": "aspect"}, "x": {"type": "ordinal", "field": "Sentiment"}, "y": {"type": "quantitative", "field": "count"}}, "title": "Aspect extracted from product reviews", "width": 60, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-16f6c0c0f8c4d4dad5efcc988033684b": [{"aspect": "", "Sentiment": 0, "count": 3707}, {"aspect": "", "Sentiment": 1, "count": 4939}, {"aspect": "coffee", "Sentiment": 0, "count": 467}, {"aspect": "coffee", "Sentiment": 1, "count": 405}, {"aspect": "dog", "Sentiment": 0, "count": 435}, {"aspect": "dog", "Sentiment": 1, "count": 512}, {"aspect": "dogs", "Sentiment": 0, "count": 221}, {"aspect": "dogs", "Sentiment": 1, "count": 366}, {"aspect": "flavor", "Sentiment": 0, "count": 668}, {"aspect": "flavor", "Sentiment": 1, "count": 542}, {"aspect": "one", "Sentiment": 0, "count": 387}, {"aspect": "one", "Sentiment": 1, "count": 321}, {"aspect": "price", "Sentiment": 0, "count": 323}, {"aspect": "price", "Sentiment": 1, "count": 583}, {"aspect": "product", "Sentiment": 0, "count": 798}, {"aspect": "product", "Sentiment": 1, "count": 583}, {"aspect": "taste", "Sentiment": 0, "count": 703}, {"aspect": "taste", "Sentiment": 1, "count": 375}, {"aspect": "tea", "Sentiment": 0, "count": 252}, {"aspect": "tea", "Sentiment": 1, "count": 426}]}}, {"mode": "vega-lite"});
</script>



What are the top most descriptors of sentiment in amazon food reviews
```python
top_descriptions = df['description'].value_counts().head(10).index
temp2 = df[df['description'].isin(top_descriptions)].groupby(['description', 'Sentiment']).size().reset_index(name='count')
alt.Chart(temp2, title='Descriptions extracted from product reviews').mark_bar().encode(
    alt.X('Sentiment:O'),
    alt.Y("count"),
    color='Sentiment:N',
    column='description'
).properties(width=60)
```





<div id="altair-viz-61edb47dae0a4d3c97aa23123f5fdb28"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-61edb47dae0a4d3c97aa23123f5fdb28") {
      outputDiv = document.getElementById("altair-viz-61edb47dae0a4d3c97aa23123f5fdb28");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-675f74243adeff1cc37df99a9531e7e8"}, "mark": "bar", "encoding": {"color": {"type": "nominal", "field": "Sentiment"}, "column": {"type": "nominal", "field": "description"}, "x": {"type": "ordinal", "field": "Sentiment"}, "y": {"type": "quantitative", "field": "count"}}, "title": "Descriptions extracted from product reviews", "width": 60, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-675f74243adeff1cc37df99a9531e7e8": [{"description": "", "Sentiment": 0, "count": 440}, {"description": "", "Sentiment": 1, "count": 329}, {"description": "best", "Sentiment": 0, "count": 162}, {"description": "best", "Sentiment": 1, "count": 545}, {"description": "better", "Sentiment": 0, "count": 432}, {"description": "better", "Sentiment": 1, "count": 264}, {"description": "free", "Sentiment": 0, "count": 189}, {"description": "free", "Sentiment": 1, "count": 347}, {"description": "good", "Sentiment": 0, "count": 879}, {"description": "good", "Sentiment": 1, "count": 1107}, {"description": "great", "Sentiment": 0, "count": 405}, {"description": "great", "Sentiment": 1, "count": 1739}, {"description": "more", "Sentiment": 0, "count": 508}, {"description": "more", "Sentiment": 1, "count": 679}, {"description": "other", "Sentiment": 0, "count": 663}, {"description": "other", "Sentiment": 1, "count": 607}, {"description": "same", "Sentiment": 0, "count": 329}, {"description": "same", "Sentiment": 1, "count": 160}, {"description": "worth", "Sentiment": 0, "count": 302}, {"description": "worth", "Sentiment": 1, "count": 255}]}}, {"mode": "vega-lite"});
</script>



Let's look at the confusion matrix of our naive classifier
```python
data = confusion_matrix(df['Sentiment'], df['predicted_sentiment'])
df_cm = pd.DataFrame(data, columns=np.unique(df['Sentiment']), index = np.unique(df['predicted_sentiment']))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,8))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
```

![png](/assets/aspect_confusion.png)


### Can we do better aspect and sentiment extraction?

Our Naive aspect extractor was not able to classify aspect of lot of reviews. We will try to imporve it by using better dependency parsing logic and using custom positive and negative dict for getting sentiment.

```python
neg_file = open("neg_words.txt",encoding = "ISO-8859-1")
pos_file = open("pos_words.txt",encoding = "ISO-8859-1")
neg = [line.strip() for line in neg_file.readlines()]
pos = [line.strip() for line in pos_file.readlines()]
opinion_words = neg + pos

def feature_sentiment(sentence):
    '''
    input: dictionary and sentence
    function: appends dictionary with new features if the feature did not exist previously,
              then updates sentiment to each of the new or existing features
    output: updated dictionary
    '''

    sent_dict = Counter()
    sentence = nlp(sentence)
    debug = 0
    for token in sentence:
    #    print(token.text,token.dep_, token.head, token.head.dep_)
        # check if the word is an opinion word, then assign sentiment
        if token.text in opinion_words:
            sentiment = 1 if token.text in pos else -1
            # if target is an adverb modifier (i.e. pretty, highly, etc.)
            # but happens to be an opinion word, ignore and pass
            if (token.dep_ == "advmod"):
                continue
            elif (token.dep_ == "amod"):
                sent_dict[token.head.text] += sentiment
            # for opinion words that are adjectives, adverbs, verbs...
            else:
                for child in token.children:
                    # if there's a adj modifier (i.e. very, pretty, etc.) add more weight to sentiment
                    # This could be better updated for modifiers that either positively or negatively emphasize
                    if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in opinion_words):
                        sentiment *= 1.5
                    # check for negation words and flip the sign of sentiment
                    if child.dep_ == "neg":
                        sentiment *= -1
                for child in token.children:
                    # if verb, check if there's a direct object
                    if (token.pos_ == "VERB") & (child.dep_ == "dobj"):                        
                        sent_dict[child.text] += sentiment
                        # check for conjugates (a AND b), then add both to dictionary
                        subchildren = []
                        conj = 0
                        for subchild in child.children:
                            if subchild.text == "and":
                                conj=1
                            if (conj == 1) and (subchild.text != "and"):
                                subchildren.append(subchild.text)
                                conj = 0
                        for subchild in subchildren:
                            sent_dict[subchild] += sentiment

                # check for negation
                for child in token.head.children:
                    noun = ""
                    if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in opinion_words):
                        sentiment *= 1.5
                    # check for negation words and flip the sign of sentiment
                    if (child.dep_ == "neg"): 
                        sentiment *= -1
                
                # check for nouns
                for child in token.head.children:
                    noun = ""
                    if (child.pos_ == "NOUN") and (child.text not in sent_dict):
                        noun = child.text
                        # Check for compound nouns
                        for subchild in child.children:
                            if subchild.dep_ == "compound":
                                noun = subchild.text + " " + noun
                        sent_dict[noun] += sentiment
                    debug += 1
    return dict(sent_dict)
df['results2'] = df['cleaned_text'].apply(lambda x: feature_sentiment(x))
```

Overall Aspect word cloud
```python
def getList(dict):
    return dict.keys()

result = []
for key, value in df['results2'].iteritems():
    keys = getList(value)
    for i in list(keys):
        result.append(i)
        
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
wordcloud = WordCloud().generate(' '.join(result))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![png](/assets/aspect_all.png)


Aspects of reviews with positive sentiment


![png](/assets/aspect_pos.png)


Aspect of reviews with negative sentiment


![png](/assets/aspect_neg.png)


Now we have better grasp of customer sentiment with granular aspects of reviews. We can make better informed decisions.

![png](/assets/aspect_final.png)


There is still a lot to be done here regarding sentiment explaination or maybe building a better classifier. That will be a post for future.





