# The Drink So Nice They Named It Twice

![whiskey-glass](images/whiskey-glass.jpg?raw=true)

Can machine learning tell the difference between a Bourbon and a Scotch? Between a whiskey and a whisky? Can it do this within the context of discussions about various types of whiskey? We will apply machine learning and modeling techniques to data from Reddit.com to find out.

Reddit is one of the most popular sites on the internet with over 11 million daily active users. The variety and breadth of subreddits allows ample opportunities to explore data collection and machine learning techniques. Here we will collect the submissions from three subreddits and build a variety of classifier models in an effort to predict from which subreddit a post is from.

### Subreddits

In order to build a rigorous model subreddits with overlapping themes were chosen.
- r/whiskey
- r/scotch
- r/bourbon

#### It's important to note that:

1. Not all whiskies are Bourbon.

2. Not all whiskies are Scotch.

3. All Scotches and Bourbons are whiskies.

The following models will be compared to see if any can beat a baseline calculation of prediction.
- Random Forest Classifier
- Multinomial Naive Bayes
- Ada Boost Classifier
- GradientBoostingClassifier

### Data Collection

Data was collected from reddit.com using the pushshift api.

https://github.com/pushshift/api

Data was collected using the reddit-subbmissions-pull.py file in the [code](./code) folder of this repository. For the purposes of this exercise five years of data was collected from 2014 to 2019.

The pushshift api returns a large variety of meta data from each submission. For the purposes of comparing natural language processing  between classification models the following features were extracted.

- title
- selftext
- score
- num_comments
- subreddit

Even though score and number of comments are numerical sets of data they are an intrensical aspect of how a subreddit may operate. How active a subreddit is via it's karma score and general discussion could be used as a way to infer the the origin of a post.

### Data Dictionary

| Variable           | Variable Name | Data Type                  | Description                                                                                                                              |
|--------------------|---------------|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Title              | title         | str                        | Tite of the post. This is what will appear on the subreddit and is mandatory                                                             |
| Post Description   | selftext      | str                        | A user can provide a description for the post that elaborates on the point or is to begin a discussion. This is not required for a post. |
| Number of Comments | num_comments  | int                        | Number of comments in the post thread. Submission does not count as comment.                                                             |
| Score              | score         | int (positive or negative) | Reddit users and upvote or downvote a post based on its relevance or general enjoyment.                                                  |
| Subreddit          | subreddit     | str                        | The subreddit the submission is from.                                                                                                    |

#### Reasons for lack of stemming and lemmatization.
Whiskeys are primarily known by their type and distillery. Given the wide range of names for Scottish distilleries there was concern that applying a stemming or lemmatization transformation could have a negative effect on model performance.

Examples:
 - Glentauchers, Mulben
 - Laphroaig, Port Ellena
 - Benrinnes, Banffshire

Additionally both vectorizers will have a the n-gram range set to (1,2) in the hyperparameter search to account for distilleries with multiples words for their names or locations.

### Model Processing

Each of the models was run with similar hyperparameters where applicable.

Additionally each model will be transformed by both a CountVectorizer and TFIDFVectorizer.

### Results
|    | Model                     | Transformer     |   Training Score |   Testing Score | Best Parameters                                                                                                                                                                                                                                                                  |   Run Time (min) |
|---:|:--------------------------|:----------------|-----------------:|----------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------:|
|  0 | Random Forest             | CountVectorizer |         0.985955 |        0.753489 | {'cvec__max_df': 0.9, 'cvec__max_features': 5000, 'cvec__min_df': 2, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': ['stopwordplaceholder'], 'cvec__strip_accents': 'unicode', 'rf__max_depth': None, 'rf__n_estimators': 100, 'rf__n_jobs': 6}                                |               48 |
|  1 | Random Forest             | TFIDFVectorizer |         0.98589  |        0.757683 | {'rf__max_depth': None, 'rf__n_estimators': 100, 'rf__n_jobs': 6, 'tfidf__max_df': 0.9, 'tfidf__max_features': 5000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 1), 'tfidf__stop_words': ['stopwordplaceholder'], 'tfidf__strip_accents': 'unicode'}                          |               45 |
|  2 | Multinomial Naive Bayes   | CountVectorizer |         0.761107 |        0.747395 | {'cvec__max_df': 0.9, 'cvec__max_features': 5000, 'cvec__min_df': 2, 'cvec__ngram_range': (1, 2), 'cvec__stop_words': ['stopwordplaceholder'], 'cvec__strip_accents': 'unicode', 'mnb__alpha': 0.01}                                                                             |                3 |
|  3 | Multinomial Naive Bayes   | TFIDFVectorizer |         0.761041 |        0.743464 | {'mnb__alpha': 0.01, 'tfidf__max_df': 0.9, 'tfidf__max_features': 5000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 2), 'tfidf__stop_words': ['stopwordplaceholder'], 'tfidf__strip_accents': 'unicode'}                                                                       |                3 |
|  4 | Ada Boost Classifier      | CountVectorizer |         0.771504 |        0.761418 | {'ada__base_estimator__max_depth': 2, 'ada__learning_rate': 0.9, 'ada__n_estimators': 150, 'cvec__max_df': 0.9, 'cvec__max_features': 3000, 'cvec__min_df': 2, 'cvec__ngram_range': (1, 2), 'cvec__stop_words': ['stopwordplaceholder'], 'cvec__strip_accents': 'unicode'}       |               81 |
|  5 | Ada Boost Classifier      | TFIDFVectorizer |         0.771504 |        0.76168  | {'ada__base_estimator__max_depth': 2, 'ada__learning_rate': 0.9, 'ada__n_estimators': 150, 'tfidf__max_df': 0.9, 'tfidf__max_features': 3000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 2), 'tfidf__stop_words': ['stopwordplaceholder'], 'tfidf__strip_accents': 'unicode'} |               81 |
|  6 | Gradient Boost Classifier | CountVectorizer |         0.800708 |        0.774196 | {'cvec__max_df': 0.9, 'cvec__max_features': 3000, 'cvec__min_df': 2, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': ['stopwordplaceholder'], 'cvec__strip_accents': 'unicode', 'gb__learning_rate': 1, 'gb__max_depth': 2, 'gb__n_estimators': 150}                            |               68 |
|  7 | Gradient Boost Classifier | TFIDFVectorizer |         0.799113 |        0.774523 | {'gb__learning_rate': 0.9, 'gb__max_depth': 2, 'gb__n_estimators': 150, 'tfidf__max_df': 0.9, 'tfidf__max_features': 3000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 1), 'tfidf__stop_words': ['stopwordplaceholder'], 'tfidf__strip_accents': 'unicode'}                    |               68 |

### Recommendations and Next Steps

#### Alter Stopwords Dictionary
- The default 'english' stopwords dictionary contains the word 'still'. I considered this to be problematic for comparing various whiskeys and their manufacturers.

#### Combine title and self text
- Expand on data set by combining the 'title' and 'self text' data. Use the 'stopwordplaceholder' as a way to prevent nulls. Adding this string to the edited stopwords dictionary will remove it from analysis.

#### Expand on Hyperparameters using AWS
- Given the process limitations of my local machine I was limited to the hyperparameters I could gridsearch across all my models. Spinning up a virtual machine to expand processing power will allow for larger and more powerful grid searches.

#### Fine Tune Pipelines
- After Expanded GridSearching a better idea of what number of features that will be ideal will be known. Pulling out the most influential keywords for comparison may show additional insights.

#### Compare just r/scotch and r/bourbon
- Removing r/whiskey data from the comparison may show larger differences in model scores
