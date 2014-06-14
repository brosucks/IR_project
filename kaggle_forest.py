import nltk.classify.util
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import csv
 
def word_feats(words):
    return dict([(word, True) for word in words])
 
if __name__ == "__main__":
  loc_train = "train.csv"
  loc_test = "test.csv"
  loc_submission = "python_submission.csv"
'''
with open('test.tsv/test.tsv','rb') as tsvin, open('test.csv', 'wb') as csvout:
    tsvin = csv.reader(tsvin, delimiter='\t')
    csvout = csv.writer(csvout)

    for row in tsvin:
        csvout.writerows([row])
'''
df_train = pd.read_csv(loc_train)
df_test = pd.read_csv(loc_test)

Phrase = df_train['Phrase']
Phrase_test = df_test['Phrase']

#negids = movie_reviews.fileids('neg')
#posids = movie_reviews.fileids('pos')
''' 
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4
''' 
trainfeats = Phrase[1:1000]
testfeats = Phrase[1000:1500]

print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)
'''
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()
'''
