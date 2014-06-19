import nltk.classify.util
import pandas as pd
import numpy as np
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import csv
 
def word_feats(words):
    return dict([(word, True) for word in words])

def find_bigrams(input_list):
    bigram_list = []
    for i in range(len(input_list)-1):
        bigram_list.append((input_list[i], input_list[i+1]))
    return bigram_list
 
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
Phrase = ['']
Phrase_label = ['']
data = open('train.csv','r')
for row in csv.DictReader(data):
    Phrase.append(row['Phrase'])
    Phrase_label.append(row['Sentiment'])
data.close()

Phrase = Phrase[1:len(Phrase)]
Phrase_label = Phrase_label[1:len(Phrase_label)]
Phrase_label = [int(numeric_string) for numeric_string in Phrase_label]

Cases = len(Phrase)

Phrase_test = ['']
Phrase_label_test = ['']
data = open('test.csv','r')
for row in csv.DictReader(data):
    Phrase_test.append(row['Phrase'])
data.close()

Phrase_test = Phrase_test[1:len(Phrase_test)]
''' 
df_train = pd.DataFrame(pd.read_csv(loc_train))
'''
df_test = pd.DataFrame(pd.read_csv(loc_test))

positive = map(lambda i : i>2 , Phrase_label[0:Cases])

negetive = map(lambda i : i<=2 , Phrase_label[0:Cases])

pos_feats = ['']
neg_feats = ['']
for row in range(Cases):
    if(positive[row]==True):
        pos_feats.append(Phrase[row])
    else:
        neg_feats.append(Phrase[row])
pos_feats = pos_feats[1:len(pos_feats)]
neg_feats = neg_feats[1:len(neg_feats)]
'''
#negids = movie_reviews.fileids('neg')
#posids = movie_reviews.fileids('pos')
'''
#str(neg_feats[k]).split() +
negfeats = [(word_feats(  find_bigrams( str(neg_feats[k]).split() ) ), '2') for k in range(0,len(neg_feats),1)]
posfeats = [(word_feats(  find_bigrams( str(pos_feats[k]).split() ) ), '3') for k in range(0,len(pos_feats),1)]
# Real test cases , 'pop' means nothing
#testfeats = [(word_feats( str(Phrase_test[k]).split()), 'pop') for k in range(0,len(Phrase_test),1)]

#neucutoff = len(neufeats)*4/5
negcutoff = len(negfeats)*4/5
poscutoff = len(posfeats)*4/5

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff] #+ neufeats[:neucutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:] #+ neufeats[neucutoff:Cases]

print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)
#classifier = nltk.classify.DecisionTreeClassifier.train(trainfeats)
#classifier = SklearnClassifier(BernoulliNB()).train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()

#results = classifier.batch_classify([fs for (fs,l) in testfeats])

count = 0
'''
with open(loc_submission, "wb") as outfile:
    outfile.write("PhraseID,Sentiment\n")
    for val in results:
      outfile.write("%s,%s\n"%(df_test['PhraseId'][count],val))
      count += 1
'''
