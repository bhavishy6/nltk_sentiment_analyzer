import collections
import itertools
import nltk.classify.util
from nltk.metrics import *
from nltk.collocations import BigramCollocationFinder
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords

stopset = set(stopwords.words('english'))

###Feature Extraction methods

#[Bigram Feature Extraction]: BigramCollocationFinder scores bigrams based on a BigramAssocMeasures(def=chi_sq) based on frequency of the bigram vs frequency of each word in the bigram
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    #find the best n bigrams based on the score_fn
    bigrams = bigram_finder.nbest(score_fn, n)
    #itertools.chain(words, bigrams) will iterate through wordset first then bigrams. Therefore 'ngram'
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

#[Stopword Filtered Feature Extraction]: basic feature extraction based on words with stopwords filtered out
def stopset_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])

#[Basic Feature Extraction]: basic feature extraction on only words.
def word_feats(words):
    return dict([(word, True) for word in words])

###Evaluate the NaiveBayesClassifier based on a Feature Extraction method.
def evaulate_classifier(featx):
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')

    negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    negcutoff = len(negfeats)*3/4
    poscutoff = len(posfeats)*3/4

    #training set of features (dictionary)
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    #testing set
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    #iterate through the dictionary
    for i, (feats, label) in enumerate(testfeats):
        #refsets (i.e.: 'neg' : )
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)


    print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    #how likely an entry in pos set is to be correct
    print 'POS precision:', precision(refsets['pos'], testsets['pos'])
    #higher recall = less false positives(in this case false negatives because it is a set of positives)
    print 'POS recall:', recall(refsets['pos'], testsets['pos'])
    #f_measure is the combination of the two metrics above
    print 'POS f_measure:', f_measure(refsets['pos'], testsets['pos'])
    #how likely the an entry in neg set is to be correct
    print 'NEG precision:', precision(refsets['neg'], testsets['neg'])
    #100-(higher recall) = % of false negatives in the neg set
    print 'NEG recall:', recall(refsets['neg'], testsets['neg'])
    #combination of precision and recall
    print 'NEG f_measure:', f_measure(refsets['neg'], testsets['neg'])

    #these most informative features show the top 'ngrams' that will factor into whether a review is positive or negative. (based on pos:neg)
    classifier.show_most_informative_features()

evaulate_classifier(bigram_word_feats)
