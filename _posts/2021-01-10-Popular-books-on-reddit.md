---
layout: post
title: "Analysis of Popular books on Reddit"
author: "Ankit"
tags: viz 
excerpt_separator: <!--more-->
---

## Only read the blog post if books interest you.<!--more-->


### I came across [reddit favourites](https://redditfavorites.com/books) which has most popular boooks from reddit that gets recommended on the platform. I wanted to see which kind of books get recommended the most, which books has highest recommendations etc. Also, I wanted to keep this handy for if i want to look for inspiration for book from somewhere i can refer to this quickly.

Basic Imports
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import altair as alt
```

Simple function to extract title, popularity_score, number of mentions of the book in reddit comments, average upvotes of the comments containing book title.
```python
def extractor(page_number):
    url = 'https://redditfavorites.com/books?page=' + str(page_number)
    x = requests.get(url)
    soup = BeautifulSoup(x.content, 'html.parser')
    # Gather tags
    title_tags = soup.find_all('h2', {'class': 'title'})
    author_tags = soup.find_all('div', {'class': 'subtitle'})
    mention_tags = soup.find_all('span', {'class': 'color-purple'})
    
    # Parsing collected tags for extracts
    title_content = []
    for tag in title_tags:
        title_content.append(tag.text)
    
    titles = title_content[::4]
    popularity_scores = title_content[1::4]

    authors = []
    for tag in author_tags:
        authors.append(tag.text)
    
    mentions_content = []
    for tag in mention_tags:
        mentions_content.append(tag.text)
    
    mentions_content = mentions_content[1:]
    comments = mentions_content[1::3]
    average_upvotes = mentions_content[2::3]
    
    # append to master list
    titles_x.append(titles)
    popularity_scores_x.append(popularity_scores)
    authors_x.append(authors)
    comments_x.append(comments)
    average_upvotes_x.append(average_upvotes)

titles_x = []
popularity_scores_x = []
authors_x = []
comments_x = []
average_upvotes_x = []

for i in [1,2]:
    extractor(i)
```

Flattening extracted attributes and create a dataframe for books and cleaning of columns.
```python
flat_titles = [item for sublist in titles_x for item in sublist]
flat_popularity_scores = [item for sublist in popularity_scores_x for item in sublist]
flat_authors = [item for sublist in authors_x for item in sublist]
flat_comments = [item for sublist in comments_x for item in sublist]
flat_average_upvotes = [item for sublist in average_upvotes_x for item in sublist]

columns = ['title', 'popularity_score', 'author', 'comments_recieved', 'average_upvotes']
books = pd.DataFrame(columns=columns)

# fill the dataframe with extracted values
books['title'] = flat_titles
books['popularity_score'] = flat_popularity_scores
books['author'] = flat_authors
books['comments_recieved'] = flat_comments
books['average_upvotes'] = flat_average_upvotes

# clean columns
books['title'] = books['title'].apply(lambda x: x[1:-1].lower())
books['popularity_score'] = books['popularity_score'].apply(lambda x: x[19:-1])
books['author'] = books['author'].apply(lambda x: x.lower())
books['average_upvotes'] = books['average_upvotes'].astype(float)
```

As the site does not provide cateory tag with the book card i extracted categories separately from categories urls and the merge with the main dataframe
```python
# book category extraction
categories = ['business', 'design', 'drawing', 'economics', 'investing', 'meditation', 'people', 'personal_finance',
             'philosophy', 'programming', 'self_improvement', 'writing']

book_category = []

def book_category_extractor(category):
    url = 'https://redditfavorites.com/books?category_id=' + str(category)
    x = requests.get(url)
    soup = BeautifulSoup(x.content, 'html.parser')
    # Gather tags
    title_tags = soup.find_all('h2', {'class': 'title'})
    title_content = []
    for tag in title_tags:
        title_content.append(tag.text)
    titles = title_content[::4]
    for title in titles:
        book_category.append([title[1:-1].lower(),category])
    
for cat in categories:
    book_category_extractor(cat)
    
columns = ['book', 'category']
df_category = pd.DataFrame(columns=columns)

book = []
category = []
for combo in book_category:
    book.append(combo[0])
    category.append(combo[1])
    
df_category['book'] = book
df_category['category'] = category

book_with_category = books.merge(df_category, left_on='title', right_on='book', how='left')

book_with_category = book_with_category.drop('book', axis=1)
```

#### What are the 20 Most recommended books on Reddit ?
```python
top_20 = book_with_category.iloc[:20,:]
alt.Chart(top_20, title='Top 20 popular books on Reddit').mark_bar().encode(
    x='popularity_score:Q',
    y="title",
    tooltip = "author"
).properties(width=600)
```





<div id="altair-viz-2b519a86c8c84a0892385cd2c330c789"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-2b519a86c8c84a0892385cd2c330c789") {
      outputDiv = document.getElementById("altair-viz-2b519a86c8c84a0892385cd2c330c789");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-0e2e2706fdb994a73a9f576198c47ed6"}, "mark": "bar", "encoding": {"tooltip": {"type": "nominal", "field": "author"}, "x": {"type": "quantitative", "field": "popularity_score"}, "y": {"type": "nominal", "field": "title"}}, "title": "Top 20 popular books on Reddit", "width": 600, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-0e2e2706fdb994a73a9f576198c47ed6": [{"title": "the art of war", "popularity_score": "8827", "author": "sun tzu", "comments_recieved": "11730", "average_upvotes": 8.42, "category": "philosophy"}, {"title": "on writing", "popularity_score": "5400", "author": "stephen king", "comments_recieved": "8005", "average_upvotes": 5.83, "category": "writing"}, {"title": "zen and the art of motorcycle maintenance", "popularity_score": "4293", "author": "robert m pirsig", "comments_recieved": "5675", "average_upvotes": 3.73, "category": "philosophy"}, {"title": "man's search for meaning", "popularity_score": "2868", "author": "viktor e. frankl", "comments_recieved": "4044", "average_upvotes": 5.98, "category": "people"}, {"title": "the intelligent investor", "popularity_score": "2434", "author": "benjamin graham, jason zweig", "comments_recieved": "3697", "average_upvotes": 4.69, "category": "investing"}, {"title": "the wealth of nations ", "popularity_score": "2385", "author": "adam smith", "comments_recieved": "3257", "average_upvotes": 11.54, "category": "economics"}, {"title": "the power of habit", "popularity_score": "2326", "author": "charles duhigg", "comments_recieved": "3301", "average_upvotes": 5.07, "category": "self_improvement"}, {"title": "think and grow rich", "popularity_score": "1949", "author": "napoleon hill, ben holden-crowther", "comments_recieved": "2703", "average_upvotes": 3.21, "category": "personal_finance"}, {"title": "the millionaire next door", "popularity_score": "1932", "author": "thomas j. stanley, william d. danko", "comments_recieved": "2955", "average_upvotes": 5.51, "category": "personal_finance"}, {"title": "clean code", "popularity_score": "1764", "author": "robert c. martin", "comments_recieved": "2751", "average_upvotes": 4.18, "category": "programming"}, {"title": "mindfulness in plain english", "popularity_score": "1763", "author": "bhante henepola gunaratana", "comments_recieved": "3442", "average_upvotes": 2.7, "category": "meditation"}, {"title": "the elements of style, fourth edition", "popularity_score": "1620", "author": "william strunk jr., e. b. white", "comments_recieved": "2094", "average_upvotes": 3.74, "category": "writing"}, {"title": "meditations ", "popularity_score": "1500", "author": "marcus aurelius", "comments_recieved": "1894", "average_upvotes": 5.5, "category": "philosophy"}, {"title": "predictably irrational, revised and expanded edition", "popularity_score": "1456", "author": "dr. dan ariely", "comments_recieved": "2125", "average_upvotes": 5.41, "category": "economics"}, {"title": "models", "popularity_score": "1453", "author": "mark manson", "comments_recieved": "2383", "average_upvotes": 3.23, "category": "self_improvement"}, {"title": "economics in one lesson", "popularity_score": "1451", "author": "henry hazlitt", "comments_recieved": "2788", "average_upvotes": 2.96, "category": "economics"}, {"title": "thinking, fast and slow", "popularity_score": "1445", "author": "daniel kahneman", "comments_recieved": "2092", "average_upvotes": 7.13, "category": "self_improvement"}, {"title": "the war of art", "popularity_score": "1391", "author": "steven pressfield", "comments_recieved": "1952", "average_upvotes": 3.49, "category": "self_improvement"}, {"title": "nicomachean ethics", "popularity_score": "1371", "author": "aristotle", "comments_recieved": "2265", "average_upvotes": 5.79, "category": "philosophy"}, {"title": "the total money makeover", "popularity_score": "1368", "author": "dave ramsey", "comments_recieved": "2464", "average_upvotes": 3.88, "category": "personal_finance"}]}}, {"mode": "vega-lite"});
</script>



#### What category of books dominate reddit recommendations ?
```python
alt.Chart(book_with_category.category.value_counts().reset_index(), title='Popular reddit books by category').mark_bar().encode(
    x='category',
    y="index",
).properties(width=600)
```





<div id="altair-viz-28daf42ea02f4bf6bcccbcacde28b541"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-28daf42ea02f4bf6bcccbcacde28b541") {
      outputDiv = document.getElementById("altair-viz-28daf42ea02f4bf6bcccbcacde28b541");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-f5e4693726bbc3b2f8a18f1c2dc553fe"}, "mark": "bar", "encoding": {"x": {"type": "quantitative", "field": "category"}, "y": {"type": "nominal", "field": "index"}}, "title": "Popular reddit books by category", "width": 600, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-f5e4693726bbc3b2f8a18f1c2dc553fe": [{"index": "programming", "category": 24}, {"index": "self_improvement", "category": 17}, {"index": "business", "category": 17}, {"index": "people", "category": 17}, {"index": "investing", "category": 16}, {"index": "economics", "category": 14}, {"index": "meditation", "category": 13}, {"index": "writing", "category": 12}, {"index": "design", "category": 12}, {"index": "drawing", "category": 11}, {"index": "personal_finance", "category": 10}, {"index": "philosophy", "category": 10}]}}, {"mode": "vega-lite"});
</script>



### Popular books by category on popularity and upvote scales
```python
alt.Chart(book_with_category, title='Popular books by category').mark_circle(size=60).encode(
    alt.X('popularity_score:Q', scale=alt.Scale(type='log', base=10)),
    alt.Y('average_upvotes:Q', scale=alt.Scale(type='log', base=10)),
    color='category',
    tooltip=['title']
).properties(width=800).interactive()
```





<div id="altair-viz-0a210631632f4d4889305c72df4418b0"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-0a210631632f4d4889305c72df4418b0") {
      outputDiv = document.getElementById("altair-viz-0a210631632f4d4889305c72df4418b0");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-9eff4d6c48c7be2c9ba20e92005ed9ee"}, "mark": {"type": "circle", "size": 60}, "encoding": {"color": {"type": "nominal", "field": "category"}, "tooltip": [{"type": "nominal", "field": "title"}], "x": {"type": "quantitative", "field": "popularity_score", "scale": {"base": 10, "type": "log"}}, "y": {"type": "quantitative", "field": "average_upvotes", "scale": {"base": 10, "type": "log"}}}, "selection": {"selector016": {"type": "interval", "bind": "scales", "encodings": ["x", "y"]}}, "title": "Popular books by category", "width": 800, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-9eff4d6c48c7be2c9ba20e92005ed9ee": [{"title": "the art of war", "popularity_score": "8827", "author": "sun tzu", "comments_recieved": "11730", "average_upvotes": 8.42, "category": "philosophy"}, {"title": "on writing", "popularity_score": "5400", "author": "stephen king", "comments_recieved": "8005", "average_upvotes": 5.83, "category": "writing"}, {"title": "zen and the art of motorcycle maintenance", "popularity_score": "4293", "author": "robert m pirsig", "comments_recieved": "5675", "average_upvotes": 3.73, "category": "philosophy"}, {"title": "man's search for meaning", "popularity_score": "2868", "author": "viktor e. frankl", "comments_recieved": "4044", "average_upvotes": 5.98, "category": "people"}, {"title": "the intelligent investor", "popularity_score": "2434", "author": "benjamin graham, jason zweig", "comments_recieved": "3697", "average_upvotes": 4.69, "category": "investing"}, {"title": "the wealth of nations ", "popularity_score": "2385", "author": "adam smith", "comments_recieved": "3257", "average_upvotes": 11.54, "category": "economics"}, {"title": "the power of habit", "popularity_score": "2326", "author": "charles duhigg", "comments_recieved": "3301", "average_upvotes": 5.07, "category": "self_improvement"}, {"title": "think and grow rich", "popularity_score": "1949", "author": "napoleon hill, ben holden-crowther", "comments_recieved": "2703", "average_upvotes": 3.21, "category": "personal_finance"}, {"title": "the millionaire next door", "popularity_score": "1932", "author": "thomas j. stanley, william d. danko", "comments_recieved": "2955", "average_upvotes": 5.51, "category": "personal_finance"}, {"title": "clean code", "popularity_score": "1764", "author": "robert c. martin", "comments_recieved": "2751", "average_upvotes": 4.18, "category": "programming"}, {"title": "mindfulness in plain english", "popularity_score": "1763", "author": "bhante henepola gunaratana", "comments_recieved": "3442", "average_upvotes": 2.7, "category": "meditation"}, {"title": "the elements of style, fourth edition", "popularity_score": "1620", "author": "william strunk jr., e. b. white", "comments_recieved": "2094", "average_upvotes": 3.74, "category": "writing"}, {"title": "meditations ", "popularity_score": "1500", "author": "marcus aurelius", "comments_recieved": "1894", "average_upvotes": 5.5, "category": "philosophy"}, {"title": "predictably irrational, revised and expanded edition", "popularity_score": "1456", "author": "dr. dan ariely", "comments_recieved": "2125", "average_upvotes": 5.41, "category": "economics"}, {"title": "models", "popularity_score": "1453", "author": "mark manson", "comments_recieved": "2383", "average_upvotes": 3.23, "category": "self_improvement"}, {"title": "economics in one lesson", "popularity_score": "1451", "author": "henry hazlitt", "comments_recieved": "2788", "average_upvotes": 2.96, "category": "economics"}, {"title": "thinking, fast and slow", "popularity_score": "1445", "author": "daniel kahneman", "comments_recieved": "2092", "average_upvotes": 7.13, "category": "self_improvement"}, {"title": "the war of art", "popularity_score": "1391", "author": "steven pressfield", "comments_recieved": "1952", "average_upvotes": 3.49, "category": "self_improvement"}, {"title": "nicomachean ethics", "popularity_score": "1371", "author": "aristotle", "comments_recieved": "2265", "average_upvotes": 5.79, "category": "philosophy"}, {"title": "the total money makeover", "popularity_score": "1368", "author": "dave ramsey", "comments_recieved": "2464", "average_upvotes": 3.88, "category": "personal_finance"}, {"title": "the shock doctrine", "popularity_score": "1366", "author": "naomi klein", "comments_recieved": "2138", "average_upvotes": 6.08, "category": "economics"}, {"title": "code complete", "popularity_score": "1251", "author": "steve mcconnell", "comments_recieved": "1889", "average_upvotes": 3.45, "category": "programming"}, {"title": "drawing on the right side of the brain", "popularity_score": "1147", "author": "betty edwards", "comments_recieved": "1719", "average_upvotes": 5.62, "category": "drawing"}, {"title": "head first java, 2nd edition", "popularity_score": "1120", "author": "kathy sierra, bert bates", "comments_recieved": "1740", "average_upvotes": 2.54, "category": "programming"}, {"title": "surely you're joking, mr. feynman! ", "popularity_score": "1113", "author": "richard p. feynman, ralph leighton", "comments_recieved": "1443", "average_upvotes": 6.94, "category": "people"}, {"title": "a random walk down wall street", "popularity_score": "1074", "author": "burton g. malkiel", "comments_recieved": "1910", "average_upvotes": 4.85, "category": "investing"}, {"title": "c programming language, 2nd edition", "popularity_score": "1028", "author": "brian w. kernighan, dennis m. ritchie", "comments_recieved": "1536", "average_upvotes": 7.22, "category": "programming"}, {"title": "effective java ", "popularity_score": "982", "author": "joshua bloch", "comments_recieved": "1478", "average_upvotes": 3.82, "category": "programming"}, {"title": "introduction to algorithms, 3rd edition ", "popularity_score": "980", "author": "thomas h. cormen, charles e. leiserson, ronald l. rivest, clifford stein", "comments_recieved": "1283", "average_upvotes": 3.52, "category": "programming"}, {"title": "zero to one", "popularity_score": "971", "author": "peter thiel, blake masters", "comments_recieved": "1235", "average_upvotes": 4.74, "category": "business"}, {"title": "how to win friends & influence people", "popularity_score": "971", "author": "dale carnegie", "comments_recieved": "1181", "average_upvotes": 6.55, "category": "self_improvement"}, {"title": "capitalism and freedom", "popularity_score": "891", "author": "milton friedman", "comments_recieved": "1381", "average_upvotes": 4.34, "category": "economics"}, {"title": "the autobiography of malcolm x", "popularity_score": "873", "author": "malcolm x, alex haley, attallah shabazz", "comments_recieved": "1203", "average_upvotes": 8.73, "category": "people"}, {"title": "the lean startup", "popularity_score": "864", "author": "eric ries", "comments_recieved": "1111", "average_upvotes": 3.12, "category": "business"}, {"title": "the richest man in babylon", "popularity_score": "861", "author": "george s. clason", "comments_recieved": "1212", "average_upvotes": 2.9, "category": "personal_finance"}, {"title": "c++ primer ", "popularity_score": "851", "author": "stanley b. lippman, jos\u00e9e lajoie, barbara e. moo", "comments_recieved": "1755", "average_upvotes": 2.72, "category": "programming"}, {"title": "the design of everyday things", "popularity_score": "683", "author": "don norman", "comments_recieved": "835", "average_upvotes": 11.96, "category": "design"}, {"title": "structure and interpretation of computer programs - 2nd edition ", "popularity_score": "672", "author": "harold abelson, gerald jay sussman", "comments_recieved": "906", "average_upvotes": 4.25, "category": "programming"}, {"title": "your money or your life", "popularity_score": "648", "author": "vicki robin, joe dominguez, monique tilford", "comments_recieved": "1302", "average_upvotes": 5.95, "category": "personal_finance"}, {"title": "the 7 habits of highly effective people", "popularity_score": "636", "author": "stephen r. covey", "comments_recieved": "782", "average_upvotes": 5.99, "category": "self_improvement"}, {"title": "when breath becomes air", "popularity_score": "626", "author": "paul kalanithi", "comments_recieved": "829", "average_upvotes": 8.44, "category": "people"}, {"title": "the pragmatic programmer", "popularity_score": "610", "author": "andrew hunt, david thomas", "comments_recieved": "935", "average_upvotes": 4.3, "category": "programming"}, {"title": "the art of computer programming, volumes 1-4a boxed set", "popularity_score": "601", "author": "donald e. knuth", "comments_recieved": "744", "average_upvotes": 6.93, "category": "programming"}, {"title": "bird by bird", "popularity_score": "566", "author": "anne lamott", "comments_recieved": "743", "average_upvotes": 3.29, "category": "writing"}, {"title": "zen mind, beginner's mind", "popularity_score": "561", "author": "shunryu suzuki", "comments_recieved": "968", "average_upvotes": 3.14, "category": "meditation"}, {"title": "the life-changing magic of tidying up", "popularity_score": "548", "author": "marie kond\u014d", "comments_recieved": "708", "average_upvotes": 8.88, "category": "self_improvement"}, {"title": "getting to yes", "popularity_score": "537", "author": "roger fisher, william l. ury, bruce patton", "comments_recieved": "717", "average_upvotes": 7.21, "category": "business"}, {"title": "so good they can't ignore you", "popularity_score": "492", "author": "cal newport", "comments_recieved": "669", "average_upvotes": 3.85, "category": "self_improvement"}, {"title": "javascript: the good parts", "popularity_score": "454", "author": "douglas crockford", "comments_recieved": "581", "average_upvotes": 3.95, "category": "programming"}, {"title": "the mythical man-month", "popularity_score": "453", "author": "frederick p. brooks jr.", "comments_recieved": "756", "average_upvotes": 5.17, "category": "programming"}, {"title": "a history of western philosophy", "popularity_score": "453", "author": "bertrand russell", "comments_recieved": "611", "average_upvotes": 3.32, "category": "philosophy"}, {"title": "the rise of theodore roosevelt ", "popularity_score": "445", "author": "edmund morris", "comments_recieved": "608", "average_upvotes": 5.35, "category": "people"}, {"title": "full catastrophe living ", "popularity_score": "414", "author": "jon kabat-zinn", "comments_recieved": "670", "average_upvotes": 8.46, "category": "meditation"}, {"title": "head first design patterns", "popularity_score": "407", "author": "eric freeman, bert bates, kathy sierra, elisabeth robson", "comments_recieved": "547", "average_upvotes": 4.54, "category": "programming"}, {"title": "a guide to the good life", "popularity_score": "397", "author": "william b. irvine", "comments_recieved": "655", "average_upvotes": 3.55, "category": "philosophy"}, {"title": "the power of now", "popularity_score": "394", "author": "eckhart tolle", "comments_recieved": "503", "average_upvotes": 2.57, "category": "meditation"}, {"title": "unbroken", "popularity_score": "380", "author": "laura hillenbrand", "comments_recieved": "513", "average_upvotes": 2.96, "category": "people"}, {"title": "the c++ programming language ", "popularity_score": "358", "author": "bjarne stroustrup", "comments_recieved": "511", "average_upvotes": 3.47, "category": "programming"}, {"title": "republic ", "popularity_score": "351", "author": "plato", "comments_recieved": "389", "average_upvotes": 2.61, "category": "philosophy"}, {"title": "the feeling good handbook", "popularity_score": "339", "author": "david d. burns", "comments_recieved": "649", "average_upvotes": 4.85, "category": "self_improvement"}, {"title": "anne frank: the diary of a young girl", "popularity_score": "326", "author": "anne frank", "comments_recieved": "396", "average_upvotes": 46.48, "category": "people"}, {"title": "code: the hidden language of computer hardware and software", "popularity_score": "320", "author": "charles petzold", "comments_recieved": "475", "average_upvotes": 3.7, "category": "programming"}, {"title": "learned optimism", "popularity_score": "319", "author": "martin e. p. seligman", "comments_recieved": "507", "average_upvotes": 3.18, "category": "self_improvement"}, {"title": "maus. i ", "popularity_score": "309", "author": "art spiegelman", "comments_recieved": "372", "average_upvotes": 8.05, "category": "people"}, {"title": "the miracle of mindfulness", "popularity_score": "292", "author": "thich nhat hanh", "comments_recieved": "468", "average_upvotes": 3.32, "category": "meditation"}, {"title": "thinking in java ", "popularity_score": "292", "author": "bruce eckel", "comments_recieved": "380", "average_upvotes": 2.33, "category": "programming"}, {"title": "the story of philosophy", "popularity_score": "269", "author": "will durant", "comments_recieved": "385", "average_upvotes": 2.29, "category": "philosophy"}, {"title": "wherever you go, there you are", "popularity_score": "267", "author": "jon kabat-zinn", "comments_recieved": "398", "average_upvotes": 3.18, "category": "meditation"}, {"title": "tools of titans", "popularity_score": "263", "author": "timothy ferriss", "comments_recieved": "309", "average_upvotes": 2.57, "category": "self_improvement"}, {"title": "artificial intelligence: a modern approach", "popularity_score": "253", "author": "stuart russell", "comments_recieved": "323", "average_upvotes": 5.35, "category": "programming"}, {"title": "the visual display of quantitative information", "popularity_score": "249", "author": "edward r. tufte", "comments_recieved": "324", "average_upvotes": 4.38, "category": "design"}, {"title": "peopleware", "popularity_score": "249", "author": "tom demarco, tim lister", "comments_recieved": "352", "average_upvotes": 4.92, "category": "business"}, {"title": "mastering the core teachings of the buddha", "popularity_score": "240", "author": "daniel ingram", "comments_recieved": "506", "average_upvotes": 2.8, "category": "meditation"}, {"title": "waking up", "popularity_score": "239", "author": "sam harris", "comments_recieved": "301", "average_upvotes": 3.46, "category": "meditation"}, {"title": "narrative of the life of frederick douglass", "popularity_score": "239", "author": "frederick douglass", "comments_recieved": "295", "average_upvotes": 4.85, "category": "people"}, {"title": "made to stick", "popularity_score": "238", "author": "chip heath, dan heath", "comments_recieved": "348", "average_upvotes": 2.88, "category": "business"}, {"title": "learning perl", "popularity_score": "233", "author": "randal l. schwartz, brian d foy, tom phoenix", "comments_recieved": "305", "average_upvotes": 3.32, "category": "programming"}, {"title": "the 4-hour workweek", "popularity_score": "231", "author": "timothy ferriss", "comments_recieved": "308", "average_upvotes": 3.87, "category": "business"}, {"title": "the algorithm design manual", "popularity_score": "230", "author": "steven s skiena", "comments_recieved": "330", "average_upvotes": 3.18, "category": "programming"}, {"title": "the e-myth revisited", "popularity_score": "224", "author": "michael e. gerber", "comments_recieved": "342", "average_upvotes": 3.54, "category": "business"}, {"title": "the man who knew infinity", "popularity_score": "218", "author": "robert kanigel", "comments_recieved": "259", "average_upvotes": 3.86, "category": "people"}, {"title": "thinking with type, 2nd revised and expanded edition", "popularity_score": "214", "author": "ellen lupton", "comments_recieved": "270", "average_upvotes": 4.35, "category": "design"}, {"title": "mindset", "popularity_score": "212", "author": "carol s. dweck", "comments_recieved": "298", "average_upvotes": 4.46, "category": "self_improvement"}, {"title": "on writing well", "popularity_score": "211", "author": "william zinsser", "comments_recieved": "237", "average_upvotes": 2.34, "category": "writing"}, {"title": "the little book of common sense investing", "popularity_score": "209", "author": "john c. bogle", "comments_recieved": "288", "average_upvotes": 4.04, "category": "investing"}, {"title": "the worldly philosophers", "popularity_score": "202", "author": "robert l. heilbroner", "comments_recieved": "300", "average_upvotes": 3.08, "category": "economics"}, {"title": "peace is every step", "popularity_score": "200", "author": "thich nhat hanh", "comments_recieved": "293", "average_upvotes": 3.07, "category": "meditation"}, {"title": "one up on wall street", "popularity_score": "199", "author": "peter lynch", "comments_recieved": "286", "average_upvotes": 3.07, "category": "investing"}, {"title": "the writers journey", "popularity_score": "195", "author": "christopher vogler", "comments_recieved": "254", "average_upvotes": 2.94, "category": "writing"}, {"title": "python cookbook, third edition", "popularity_score": "184", "author": "david beazley, brian k. jones", "comments_recieved": "265", "average_upvotes": 3.42, "category": "programming"}, {"title": "the $100 startup", "popularity_score": "181", "author": "chris guillebeau", "comments_recieved": "249", "average_upvotes": 2.09, "category": "business"}, {"title": "the innovator's dilemma", "popularity_score": "174", "author": "clayton m. christensen", "comments_recieved": "240", "average_upvotes": 4.03, "category": "business"}, {"title": "the four pillars of investing", "popularity_score": "172", "author": "william j. bernstein", "comments_recieved": "380", "average_upvotes": 3.72, "category": "investing"}, {"title": "washington", "popularity_score": "166", "author": "ron chernow", "comments_recieved": "248", "average_upvotes": 16.03, "category": "people"}, {"title": "programming perl", "popularity_score": "159", "author": "tom christiansen, brian d foy, larry wall, jon orwant", "comments_recieved": "206", "average_upvotes": 4.16, "category": "programming"}, {"title": "common stocks and uncommon profits and other writings", "popularity_score": "154", "author": "philip a. fisher", "comments_recieved": "203", "average_upvotes": 4.42, "category": "investing"}, {"title": "keys to drawing", "popularity_score": "151", "author": "bert dodson", "comments_recieved": "248", "average_upvotes": 2.42, "category": "drawing"}, {"title": "steve jobs", "popularity_score": "151", "author": "walter isaacson", "comments_recieved": "185", "average_upvotes": 2.56, "category": "people"}, {"title": "running lean", "popularity_score": "149", "author": "ash maurya", "comments_recieved": "228", "average_upvotes": 2.34, "category": "business"}, {"title": "how to fail at almost everything and still win big", "popularity_score": "147", "author": "scott adams", "comments_recieved": "163", "average_upvotes": 5.8, "category": "self_improvement"}, {"title": "the daily stoic", "popularity_score": "140", "author": "ryan holiday, stephen hanselman", "comments_recieved": "181", "average_upvotes": 3.42, "category": "philosophy"}, {"title": "search inside yourself", "popularity_score": "138", "author": "chade-meng tan, daniel goleman, jon kabat-zinn", "comments_recieved": "212", "average_upvotes": 2.65, "category": "meditation"}, {"title": "the armchair economist", "popularity_score": "138", "author": "steven e. landsburg", "comments_recieved": "197", "average_upvotes": 3.29, "category": "economics"}, {"title": "the linux programming interface", "popularity_score": "135", "author": "michael kerrisk", "comments_recieved": "171", "average_upvotes": 4.62, "category": "programming"}, {"title": "working effectively with legacy code", "popularity_score": "135", "author": "michael feathers", "comments_recieved": "173", "average_upvotes": 3.7, "category": "programming"}, {"title": "i will teach you to be rich", "popularity_score": "134", "author": "ramit sethi", "comments_recieved": "171", "average_upvotes": 3.3, "category": "personal_finance"}, {"title": "10% happier", "popularity_score": "133", "author": "dan harris", "comments_recieved": "158", "average_upvotes": 3.03, "category": "meditation"}, {"title": "the hard thing about hard things", "popularity_score": "132", "author": "ben horowitz", "comments_recieved": "154", "average_upvotes": 3.23, "category": "business"}, {"title": "zen in the art of writing", "popularity_score": "119", "author": "ray bradbury", "comments_recieved": "150", "average_upvotes": 4.73, "category": "writing"}, {"title": "the bogleheads' guide to investing", "popularity_score": "117", "author": "taylor larimore, mel lindauer, michael leboeuf", "comments_recieved": "264", "average_upvotes": 2.78, "category": "investing"}, {"title": "deep work", "popularity_score": "114", "author": "cal newport", "comments_recieved": "132", "average_upvotes": 2.48, "category": "self_improvement"}, {"title": "the sense of style", "popularity_score": "114", "author": "steven pinker", "comments_recieved": "142", "average_upvotes": 3.31, "category": "writing"}, {"title": "universal principles of design, revised and updated", "popularity_score": "112", "author": "william lidwell, kritina holden, jill butler", "comments_recieved": "136", "average_upvotes": 2.43, "category": "design"}, {"title": "alexander hamilton", "popularity_score": "111", "author": "ron chernow", "comments_recieved": "159", "average_upvotes": 10.35, "category": "people"}, {"title": "fun with a pencil", "popularity_score": "106", "author": "andrew loomis", "comments_recieved": "154", "average_upvotes": 3.0, "category": "drawing"}, {"title": "doing good better", "popularity_score": "105", "author": "william macaskill", "comments_recieved": "166", "average_upvotes": 15.1, "category": "self_improvement"}, {"title": "drawing the head and hands", "popularity_score": "104", "author": "andrew loomis", "comments_recieved": "131", "average_upvotes": 2.37, "category": "drawing"}, {"title": "figure drawing for all it's worth", "popularity_score": "97", "author": "andrew loomis", "comments_recieved": "139", "average_upvotes": 2.19, "category": "drawing"}, {"title": "logo design love", "popularity_score": "91", "author": "david airey", "comments_recieved": "108", "average_upvotes": 3.3, "category": "design"}, {"title": "the practice of programming ", "popularity_score": "89", "author": "brian w. kernighan, rob pike", "comments_recieved": "166", "average_upvotes": 3.71, "category": "programming"}, {"title": "reading like a writer", "popularity_score": "74", "author": "francine prose", "comments_recieved": "100", "average_upvotes": 2.17, "category": "writing"}, {"title": "the affluent society", "popularity_score": "73", "author": "john  kenneth galbraith", "comments_recieved": "99", "average_upvotes": 4.45, "category": "economics"}, {"title": "the non-designer's design book ", "popularity_score": "68", "author": "robin williams", "comments_recieved": "85", "average_upvotes": 3.04, "category": "design"}, {"title": "atlas of human anatomy for the artist", "popularity_score": "65", "author": "stephen rogers peck", "comments_recieved": "74", "average_upvotes": 2.27, "category": "drawing"}, {"title": "design for hackers", "popularity_score": "63", "author": "david kadavy", "comments_recieved": "76", "average_upvotes": 3.13, "category": "design"}, {"title": "benjamin franklin", "popularity_score": "62", "author": "walter isaacson", "comments_recieved": "70", "average_upvotes": 17.56, "category": "people"}, {"title": "the alchemy of finance", "popularity_score": "60", "author": "george soros", "comments_recieved": "70", "average_upvotes": 5.84, "category": "investing"}, {"title": "pop internationalism ", "popularity_score": "60", "author": "paul krugman", "comments_recieved": "99", "average_upvotes": 6.49, "category": "economics"}, {"title": "designing brand identity", "popularity_score": "52", "author": "alina wheeler", "comments_recieved": "61", "average_upvotes": 3.61, "category": "design"}, {"title": "smarter faster better", "popularity_score": "51", "author": "charles duhigg", "comments_recieved": "56", "average_upvotes": 3.21, "category": "self_improvement"}, {"title": "the undercover economist strikes back", "popularity_score": "50", "author": "tim harford", "comments_recieved": "98", "average_upvotes": 4.4, "category": "economics"}, {"title": "get a financial life", "popularity_score": "50", "author": "beth kobliner", "comments_recieved": "122", "average_upvotes": 2.2, "category": "personal_finance"}, {"title": "elon musk", "popularity_score": "50", "author": "ashlee vance", "comments_recieved": "71", "average_upvotes": 2.04, "category": "people"}, {"title": "the essays of warren buffett", "popularity_score": "43", "author": "warren e. buffett, lawrence a. cunningham", "comments_recieved": "47", "average_upvotes": 2.96, "category": "investing"}, {"title": "bridgman's complete guide to drawing from life", "popularity_score": "42", "author": "george b. bridgman", "comments_recieved": "52", "average_upvotes": 2.71, "category": "drawing"}, {"title": "hedge fund market wizards", "popularity_score": "42", "author": "jack d. schwager", "comments_recieved": "66", "average_upvotes": 2.5, "category": "investing"}, {"title": "titan", "popularity_score": "40", "author": "ron chernow", "comments_recieved": "51", "average_upvotes": 6.73, "category": "people"}, {"title": "thinking strategically", "popularity_score": "40", "author": "avinash k. dixit, barry j. nalebuff", "comments_recieved": "71", "average_upvotes": 2.11, "category": "economics"}, {"title": "nudge", "popularity_score": "33", "author": "richard h. thaler, cass r. sunstein", "comments_recieved": "53", "average_upvotes": 4.72, "category": "self_improvement"}, {"title": "drawing the head and figure", "popularity_score": "30", "author": "jack hamm", "comments_recieved": "38", "average_upvotes": 1.87, "category": "drawing"}, {"title": "the pig that wants to be eaten", "popularity_score": "29", "author": "julian baggini", "comments_recieved": "31", "average_upvotes": 2.35, "category": "philosophy"}, {"title": "the accidental theorist", "popularity_score": "28", "author": "paul krugman", "comments_recieved": "52", "average_upvotes": 6.77, "category": "economics"}, {"title": "write. publish. repeat.", "popularity_score": "28", "author": "sean platt, johnny b. truant", "comments_recieved": "37", "average_upvotes": 2.84, "category": "writing"}, {"title": "the myth of the rational market", "popularity_score": "25", "author": "justin fox", "comments_recieved": "52", "average_upvotes": 3.44, "category": "investing"}, {"title": "value investing", "popularity_score": "23", "author": "bruce c. n. greenwald, judd kahn, paul d. sonkin, michael van biema", "comments_recieved": "27", "average_upvotes": 4.19, "category": "investing"}, {"title": "the little book that still beats the market", "popularity_score": "22", "author": "joel greenblatt", "comments_recieved": "26", "average_upvotes": 2.96, "category": "investing"}, {"title": "turning the mind into an ally", "popularity_score": "22", "author": "sakyong mipham", "comments_recieved": "28", "average_upvotes": 1.82, "category": "meditation"}, {"title": "good to great", "popularity_score": "21", "author": "jim collins", "comments_recieved": "22", "average_upvotes": 1.09, "category": "business"}, {"title": "the manual of ideas", "popularity_score": "18", "author": "john mihaljevic", "comments_recieved": "27", "average_upvotes": 2.07, "category": "investing"}, {"title": "growing a business", "popularity_score": "18", "author": "paul hawken", "comments_recieved": "21", "average_upvotes": 3.52, "category": "business"}, {"title": "rendering in pen and ink", "popularity_score": "17", "author": "arthur l. guptill", "comments_recieved": "42", "average_upvotes": 3.79, "category": "drawing"}, {"title": "business adventures", "popularity_score": "16", "author": "john brooks", "comments_recieved": "20", "average_upvotes": 3.4, "category": "business"}, {"title": "design as art ", "popularity_score": "16", "author": "bruno munari", "comments_recieved": "21", "average_upvotes": 2.38, "category": "design"}, {"title": "why smart people make big money mistakes and how to correct them", "popularity_score": "15", "author": "gary belsky, thomas gilovich", "comments_recieved": "18", "average_upvotes": 3.89, "category": "personal_finance"}, {"title": "micromotives and macrobehavior ", "popularity_score": "14", "author": "thomas c. schelling", "comments_recieved": "23", "average_upvotes": 3.87, "category": "economics"}, {"title": "story genius", "popularity_score": "13", "author": "lisa cron", "comments_recieved": "14", "average_upvotes": 1.71, "category": "writing"}, {"title": "designing design", "popularity_score": "13", "author": "kenya hara", "comments_recieved": "16", "average_upvotes": 3.06, "category": "design"}, {"title": "work less, live more", "popularity_score": "12", "author": "robert clyatt", "comments_recieved": "27", "average_upvotes": 7.7, "category": "personal_finance"}, {"title": "traction", "popularity_score": "11", "author": "gino wickman", "comments_recieved": "17", "average_upvotes": 2.24, "category": "business"}, {"title": "white", "popularity_score": "10", "author": "kenya hara", "comments_recieved": "11", "average_upvotes": 3.91, "category": "design"}, {"title": "built to last", "popularity_score": "10", "author": "jim collins, jerry i porras", "comments_recieved": "20", "average_upvotes": 1.0, "category": "business"}, {"title": "ernest hemingway on writing", "popularity_score": "9", "author": "", "comments_recieved": "9", "average_upvotes": 17.56, "category": "writing"}, {"title": "churchill", "popularity_score": "9", "author": "martin gilbert", "comments_recieved": "9", "average_upvotes": 2.78, "category": "people"}, {"title": "macroeconomic patterns and stories", "popularity_score": "9", "author": "edward e. leamer", "comments_recieved": "13", "average_upvotes": 3.69, "category": "economics"}, {"title": "drawing for the absolute beginner", "popularity_score": "8", "author": "mark willenbrink, mary willenbrink", "comments_recieved": "11", "average_upvotes": 2.0, "category": "drawing"}, {"title": "berkshire hathaway letters to shareholders", "popularity_score": "8", "author": "warren buffett", "comments_recieved": "8", "average_upvotes": 2.75, "category": "investing"}, {"title": "building wealth and being happy", "popularity_score": "8", "author": "graeme falco", "comments_recieved": "8", "average_upvotes": 12.88, "category": "personal_finance"}, {"title": "focused and fearless", "popularity_score": "5", "author": "shaila catherine", "comments_recieved": "11", "average_upvotes": 1.64, "category": "meditation"}, {"title": "how to use graphic design to sell things, explain things, make things look better, make people laugh, make people cry, and (every once in a while) change the world\u00a0", "popularity_score": "5", "author": "michael bierut", "comments_recieved": "5", "average_upvotes": 14.4, "category": "design"}, {"title": "the startup playbook", "popularity_score": "3", "author": "david kidder", "comments_recieved": "3", "average_upvotes": 1.67, "category": "business"}, {"title": "stein on writing", "popularity_score": "3", "author": "sol stein", "comments_recieved": "7", "average_upvotes": 1.43, "category": "writing"}, {"title": "a history of interest rates, fourth edition ", "popularity_score": "1", "author": "sidney  homer, richard sylla", "comments_recieved": "1", "average_upvotes": 1.0, "category": "investing"}, {"title": "the art of animal drawing", "popularity_score": "1", "author": "ken hultgren", "comments_recieved": "1", "average_upvotes": 1.0, "category": "drawing"}]}}, {"mode": "vega-lite"});
</script>

Being a data guy and a techie i wanted to know the popular programming books on reddit !
#### Most popular programming books on reddit

```python
alt.Chart(book_with_category.loc[book_with_category['category']=='programming'], title='Popular programming books').mark_circle(size=60).encode(
    alt.X('popularity_score:Q', scale=alt.Scale(type='log', base=10)),
    alt.Y('average_upvotes:Q', scale=alt.Scale(type='log', base=10)),
    tooltip=['title']
).properties(width=800).interactive()
```





<div id="altair-viz-20f7e15d446b46db9c618c6fc17ca382"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-20f7e15d446b46db9c618c6fc17ca382") {
      outputDiv = document.getElementById("altair-viz-20f7e15d446b46db9c618c6fc17ca382");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-e7e7a5ee4d27e8b793b738e718685cd3"}, "mark": {"type": "circle", "size": 60}, "encoding": {"tooltip": [{"type": "nominal", "field": "title"}], "x": {"type": "quantitative", "field": "popularity_score", "scale": {"base": 10, "type": "log"}}, "y": {"type": "quantitative", "field": "average_upvotes", "scale": {"base": 10, "type": "log"}}}, "selection": {"selector015": {"type": "interval", "bind": "scales", "encodings": ["x", "y"]}}, "title": "Popular programming books", "width": 800, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-e7e7a5ee4d27e8b793b738e718685cd3": [{"title": "clean code", "popularity_score": "1764", "author": "robert c. martin", "comments_recieved": "2751", "average_upvotes": 4.18, "category": "programming"}, {"title": "code complete", "popularity_score": "1251", "author": "steve mcconnell", "comments_recieved": "1889", "average_upvotes": 3.45, "category": "programming"}, {"title": "head first java, 2nd edition", "popularity_score": "1120", "author": "kathy sierra, bert bates", "comments_recieved": "1740", "average_upvotes": 2.54, "category": "programming"}, {"title": "c programming language, 2nd edition", "popularity_score": "1028", "author": "brian w. kernighan, dennis m. ritchie", "comments_recieved": "1536", "average_upvotes": 7.22, "category": "programming"}, {"title": "effective java ", "popularity_score": "982", "author": "joshua bloch", "comments_recieved": "1478", "average_upvotes": 3.82, "category": "programming"}, {"title": "introduction to algorithms, 3rd edition ", "popularity_score": "980", "author": "thomas h. cormen, charles e. leiserson, ronald l. rivest, clifford stein", "comments_recieved": "1283", "average_upvotes": 3.52, "category": "programming"}, {"title": "c++ primer ", "popularity_score": "851", "author": "stanley b. lippman, jos\u00e9e lajoie, barbara e. moo", "comments_recieved": "1755", "average_upvotes": 2.72, "category": "programming"}, {"title": "structure and interpretation of computer programs - 2nd edition ", "popularity_score": "672", "author": "harold abelson, gerald jay sussman", "comments_recieved": "906", "average_upvotes": 4.25, "category": "programming"}, {"title": "the pragmatic programmer", "popularity_score": "610", "author": "andrew hunt, david thomas", "comments_recieved": "935", "average_upvotes": 4.3, "category": "programming"}, {"title": "the art of computer programming, volumes 1-4a boxed set", "popularity_score": "601", "author": "donald e. knuth", "comments_recieved": "744", "average_upvotes": 6.93, "category": "programming"}, {"title": "javascript: the good parts", "popularity_score": "454", "author": "douglas crockford", "comments_recieved": "581", "average_upvotes": 3.95, "category": "programming"}, {"title": "the mythical man-month", "popularity_score": "453", "author": "frederick p. brooks jr.", "comments_recieved": "756", "average_upvotes": 5.17, "category": "programming"}, {"title": "head first design patterns", "popularity_score": "407", "author": "eric freeman, bert bates, kathy sierra, elisabeth robson", "comments_recieved": "547", "average_upvotes": 4.54, "category": "programming"}, {"title": "the c++ programming language ", "popularity_score": "358", "author": "bjarne stroustrup", "comments_recieved": "511", "average_upvotes": 3.47, "category": "programming"}, {"title": "code: the hidden language of computer hardware and software", "popularity_score": "320", "author": "charles petzold", "comments_recieved": "475", "average_upvotes": 3.7, "category": "programming"}, {"title": "thinking in java ", "popularity_score": "292", "author": "bruce eckel", "comments_recieved": "380", "average_upvotes": 2.33, "category": "programming"}, {"title": "artificial intelligence: a modern approach", "popularity_score": "253", "author": "stuart russell", "comments_recieved": "323", "average_upvotes": 5.35, "category": "programming"}, {"title": "learning perl", "popularity_score": "233", "author": "randal l. schwartz, brian d foy, tom phoenix", "comments_recieved": "305", "average_upvotes": 3.32, "category": "programming"}, {"title": "the algorithm design manual", "popularity_score": "230", "author": "steven s skiena", "comments_recieved": "330", "average_upvotes": 3.18, "category": "programming"}, {"title": "python cookbook, third edition", "popularity_score": "184", "author": "david beazley, brian k. jones", "comments_recieved": "265", "average_upvotes": 3.42, "category": "programming"}, {"title": "programming perl", "popularity_score": "159", "author": "tom christiansen, brian d foy, larry wall, jon orwant", "comments_recieved": "206", "average_upvotes": 4.16, "category": "programming"}, {"title": "the linux programming interface", "popularity_score": "135", "author": "michael kerrisk", "comments_recieved": "171", "average_upvotes": 4.62, "category": "programming"}, {"title": "working effectively with legacy code", "popularity_score": "135", "author": "michael feathers", "comments_recieved": "173", "average_upvotes": 3.7, "category": "programming"}, {"title": "the practice of programming ", "popularity_score": "89", "author": "brian w. kernighan, rob pike", "comments_recieved": "166", "average_upvotes": 3.71, "category": "programming"}]}}, {"mode": "vega-lite"});
</script>

I might expand on this post in future with Text analysis on to rated comments related to the books.
Interesting to see that Reddit great place for learning and supportive community for diverse interests.
