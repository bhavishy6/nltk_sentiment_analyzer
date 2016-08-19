import collections
import nltk.classify.util
from nltk.metrics import *
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords

stopset = set(stopwords.words('english'))

def word_feats(words, filterset=None):
    if filterset is None:
        return dict([(word, True) for word in words])
    else:
        return dict([(word, True) for word in words if word not in filterset])

def evaulate_classifier(featx, filterset=None):
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')

    negfeats = [(featx(movie_reviews.words(fileids=[f]), filterset), 'neg') for f in negids]
    posfeats = [(featx(movie_reviews.words(fileids=[f]), filterset), 'pos') for f in posids]

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

    classifier.show_most_informative_features()

evaulate_classifier(word_feats, stopset)
