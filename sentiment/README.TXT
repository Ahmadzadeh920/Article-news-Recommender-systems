This archive contains collection of 16327 articles from three Irish online news sources (RTE,The Irish Times,The Irish Independent) collected between 2009-07-08 and 2009-10-16. A subset of 1660 of these articles were annotated manually by 33 users during that period.

The collection consists of two datasets (user annotations and term frequencies):

Warm-up: The first month constituted a "warm-up" period, which allowed us to train the relevance classifier. This provided an initial dataset containing 3858 articles, with 2693 user annotations covering 354 individual articles.

Main: For the latter two months of the experiment, we collected a dataset for evaluating the machine learning questions arising from the sentiment analysis task. This second "main" dataset comprises 12469 documents, with 6910 user annotations resulting in 1306 labeled articles. 

A combined dataset ("all") is also provided.

Each dataset is provided as three comma-separated text files, with the following file formats:

* sentiment_*_terms.csv: Contains the list of all unique terms in the dataset vocabulary, with one line per term:
<term_id>,<term_text>

* sentiment_*_articles.csv: Contains the list of all articles in the dataset, including term index and frequency pairs.
<article_id>,<article_data(yyyy-mm-dd)>,<provider(rte,irishtimes,independent)>,<term_index1:freq1>,...,<term_indexN:freqN>

* sentiment_*_annotations.csv: Contains the list of all user annotations.
<article_id>,<user_id>,<annotation>

All rights, including copyright, in the content of the original articles are owned by the the original authors/publishers at:
http://www.rte.ie/news
http://www.independent.ie
http://www.irishtimes.com

