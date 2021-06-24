# Matrix Factorization

We come across recommendations multiple times a day — while deciding what to watch on Netflix/Youtube, item recommendations on shopping sites, song suggestions on Spotify, friend recommendations on Instagram, job recommendations on LinkedIn…the list goes on! Recommender systems aim to predict the “rating” or “preference” a user would give to an item. These ratings are used to determine what a user might like and make informed suggestions.

* There are two broad types of Recommender systems:

1. **Content-Based systems**: These systems try to match users with items based on items’ content (genre, color, etc) and users’ profiles (likes, dislikes, demographic information, etc). For example, Youtube might suggest me cooking videos based on the fact that I’m a chef, and/or that I’ve watched a lot of baking videos in the past, hence utilizing the information it has about a video’s content and my profile.

2. **Collaborative filtering** : They rely on the assumption that similar users like similar items. Similarity measures between users and/or items are used to make recommendations.

## Matrix Factorization

A recommender system has two entities — users and items. Let’s say we have `m` users and `n` items. The goal of our recommendation system is to build an mxn matrix (called the **utility matrix**) which consists of the rating (or preference) for each user-item pair. Initially, this matrix is usually very sparse because we only have ratings for a limited number of user-item pairs.

Now, our goal is to populate this matrix by finding similarities between users and items. To get an intuition, for example, we see that User3 and User4 gave the same rating to Batman, so we can assume the users are similar and they’d feel the same way about Spiderman and predict that User3 would give a rating of 4 to Spiderman. In practice, however, this is not as straightforward because there are multiple users interacting with many different items.

## Limitations of Matrix Factorization
Matrix factorization is a very simple and convenient way to make recommendations. However, it has its pitfalls, one of which we have already encountered in our implementation:

**Cold Start Problem**

We cannot make predictions for items and users we’ve never encountered in the training data because we don’t have embeddings for them.
The cold-start problem can be addressed in many ways including Recommending popular items, Asking the user to rate some items, using a content-based approach until we have enough data to use collaborative filtering.

**Hard to include additional context about user/item**

We only used user IDs and item IDs to create the embeddings. We are not able to use any other information about our users and items in our implementation. There are some complex hybrid models of content-based collaborative filtering that can be used to address this.

**Ratings are not always available**

It’s hard to get feedback from users. Most users rate things only when they really like something or absolutely hate it. In such cases, we often have to come up with a way to measure implicit feedback and use negative sampling techniques to come up with a reasonable training set.

## Conclusion
Recommendation systems are really interesting but can also get too complex too easily, especially when implemented at scale where there are millions of users and millions of items. If you want to dive deeper into recommendation systems, check out various case studies — how Youtube recommends videos, LinkedIn job recommendations.

