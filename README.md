# Matrix Factorization

We come across recommendations multiple times a day — while deciding what to watch on Netflix/Youtube, item recommendations on shopping sites, song suggestions on Spotify, friend recommendations on Instagram, job recommendations on LinkedIn…the list goes on! Recommender systems aim to predict the “rating” or “preference” a user would give to an item. These ratings are used to determine what a user might like and make informed suggestions.

* There are two broad types of Recommender systems:

1. **Content-Based systems**: These systems try to match users with items based on items’ content (genre, color, etc) and users’ profiles (likes, dislikes, demographic information, etc). For example, Youtube might suggest me cooking videos based on the fact that I’m a chef, and/or that I’ve watched a lot of baking videos in the past, hence utilizing the information it has about a video’s content and my profile.

2. **Collaborative filtering** : They rely on the assumption that similar users like similar items. Similarity measures between users and/or items are used to make recommendations.

## Matrix Factorization

A recommender system has two entities — users and items. Let’s say we have `m` users and `n` items. The goal of our recommendation system is to build an mxn matrix (called the **utility matrix**) which consists of the rating (or preference) for each user-item pair. Initially, this matrix is usually very sparse because we only have ratings for a limited number of user-item pairs.

Now, our goal is to populate this matrix by finding similarities between users and items. To get an intuition, for example, we see that User3 and User4 gave the same rating to Batman, so we can assume the users are similar and they’d feel the same way about Spiderman and predict that User3 would give a rating of 4 to Spiderman. In practice, however, this is not as straightforward because there are multiple users interacting with many different items.