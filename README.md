# nltk_sentiment_analyzer
using nltk to analyze sentiment

Using [this post](http://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/) as guidance

Options:
- [Bigram Feature Extraction]: scores bigrams based on a BigramAssocMeasures(def=chi_sq) based on frequency of the bigram vs frequency of each word in the bigram
- [Stopword Filtered Feature Extraction]: basic feature extraction based on words with stopwords filtered out
- [Basic Feature Extraction]: basic feature extraction on only words.
- [NaiveBayesClassifier] based on a Feature Extraction method


End Goal:
- Scan tweets to discover the general public opinion of a topic/product.
