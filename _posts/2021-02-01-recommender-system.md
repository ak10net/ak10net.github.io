---
layout: post
title: "Recsys part one"
author: "Ankit"
tags: recommender
excerpt_separator: <!--more-->
---
## In the series of blog post we will undertand and implement recommender system<!--more-->

# Recommender system part one - Theory

It's 2021. Let's see how much of my world is shaped by recommender system. When i wake up with my morning coffee i search google news and scroll down headlines and click which interest me. Over a period of time google news have atuned to my prefrences it knows what kind of news my interest me and it serves me thru recommendation system. If you know me, i love listening to music and love discovering new artists. Prior to spotify and youtube it was difficult to discover new music. Now, new music gets delivered to me thru excellent recommenders of spotify and youtube. I love reading books and generally rely on recommendations from people i think look up to in particular area of interest. I also like discovering good movies i usually go by imdb ratings and it's recommendation of similar movies. In short, Recommender system have lot of influence in my life. Also, they make lot of money for companies from increased sales, ctr, engagement etc.

Although recommender systems talked above can get pretty complex. We will try to understand the essense of recommendation system. What they do, how they do in simplest way possible.

**Let's get rolling**

##  Raw material required for recommender systems :- 

1. *User* - We need someone to recommend to right. So, we have data related to user.

2. *Items* - What need something to recommend to it could be anything from product on amazon, to book, song or video to users.

3. *Interactions* - We also need interactions of users and product it could be explcit interaction such as rating or implicit interaction such as purchase, view etc.

## There can be several basis for recommendations

1. *Frequency based recommendations* - Why not recommend top items to everyone think of it like youtube trending recommendations. Those are items that got most like / views in certain time period in particular geography so they are recommended to all.

2. *Recommendation based on item property* - Items that have somewhat similar attribute can be good candidate for recommendation. Like movies with same actors as user have watch or movie with similar plot. for such recommender features need to be extracted and similarity need to be scored before recommendation.

3. *Group recommendation* - Items that belong to same groups can be recommended together. If based on item properties we can cluster the items into different group we can recommend similar item to what user is looking for in the same group.

4. *Recommendations based on other user's recommendation* - Users that have somewhat similar consumption history can be recommended novelty items that they haven't interacted with but other similar user have.

There can be many more. All of the approaches have their own shortcomings. 

Since lot of background calculations for recommender systems are done with similarity metrics
It is good to know about them 
Methods to calculate similarity
1. [Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index)
2. [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)
3. [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
4. [Dot product](https://en.wikipedia.org/wiki/Dot_product)
5. [Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)


## Let's understand the types of recommender approaches

### Content based approach
- This approach is based on property or attribute of an item.
- This appraoch doesn't need any data about other users, since the recommendations are specific to this user.This makes it easier to scale to a large number of users.
- This approach can capture the specific interests of a user, and can recommend niche items that very few other users are interested in.
- Since the feature representation of the items are hand-engineered to some extent, this technique requires a lot of domain knowledge.
- This approach uses similarity between items to recommend items similar to what the user likes
- Example - If user A watches two cute cat videos, then the system can recommend cute animal videos to that user.

### Collaborative filtering approach
- This approach is based solely on past user item interaction
- This approach uses similarities between queries and items simultaneously to provide recommendations.This allows for serendipitous recommendations
- If user A is similar to user B, and user B likes video 1, then the system can recommend video 1 to user A (even if user A hasn’t seen any videos similar to video 1).
- In this approach no domain knowledge neccessary
- Interactions could be Explicit and Implicit (ratings, watch time, click etc)

*Can be Further divided into* 

####  Memory based approach (KNN etc)
- This approach used KNN and such algorithms to find out nearest users ot items 
- These do not suffer from cold start problem like model based approaches 
- There are no latent models
- There are no optimization algorithm like SGD, ALS
- This approach does not scale easily as computation becomes heavy at serving time
- Time consuming for big systems

*Memory based approach can be further divided into* 
##### User-user type
- This approach uses User features age, sex, location etc
- It Identify user with most similar interaction profile using similarity metrics
- Suggestion to a user would be average of interaction of its closest neighbors
- In short this suggest items most popular among closes neighbors of user
- This model has high variance and low bias

##### Item-item type
- This approach uses item features type, category etc
- Same as user-user but for item
- Find similar items to the ones the user already positively intercted with in the past
- Suggest K nearest neighbors to selected 'best item' that are new to our user
- This model high bias and low variance
	
#### Model based approach
- In this appraoch Recommendations are done following model information
- This appraoch require no information about user and item
- As more interactions happen more accurate recommendations becomes
- This Suffers from cold start problem hard to recommend to new user or new item till sufficient interactions are not there
- Assumes latent model SVD, NMF etc
- To calculate latent factors optmization algorithm such as SGD and ALS are used

*Model based approach can be further divided into* 
##### Matrix factorization (User-Item)
- Decompose huge and sparse matrix (m*n) into a product of two smaller and dense matrix a user-factor matrix (m*k) and a factor-item matrix (k*n)
- Matrix factorization can be done with SVD, PCA,NMF etc
- Uses gradient descent or alternating least square for matrix convergence

##### Deep learning (User-Item)
- Deep learning based models can use side features that Matrix factorization cannot

	
### Hybrid models
- Mix of content and collaborative filtering models

That is a lot about recommender systems to wrap ones head around.
Going forward in this series we will try to build few recommender system based using python.


Refrences:-
[Mining of massive datasets](http://www.mmds.org/)
[Google recommender crash course](https://developers.google.com/machine-learning/recommendation)
[Introduction to recsys theory](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada)
[Excellent blog post](https://jessesw.com/Rec-System/)
[Excellent blog post](https://datascienceplus.com/building-a-book-recommender-system-the-basics-knn-and-matrix-factorization/)
[Package](https://github.com/benfred/implicit)
[Cool blog post](https://www.benfrederickson.com/matrix-factorization/)