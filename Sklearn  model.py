import nltk.classify.util
import pandas as pd
import numpy as np
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from nltk.tokenize import RegexpTokenizer
import csv
 
def word_feats(words):
    return dict([(word, True) for word in words])

def find_bigrams(input_list):
    bigram_list = []
    for i in range(len(input_list)):
        bigram_list.append((input_list[i]))
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
tokenizer = RegexpTokenizer(r'\w+')

Phrase = ['']
Phrase_label = ['']
data = open('train.csv','r')
for row in csv.DictReader(data):
    Phrase.append(row['Phrase'].lower())
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
    Phrase_test.append(row['Phrase'].lower())
data.close()

Phrase_test = Phrase_test[1:len(Phrase_test)]


''' 
df_train = pd.DataFrame(pd.read_csv(loc_train))
'''
df_test = pd.DataFrame(pd.read_csv(loc_test))

feats_0 = ['']
feats_1 = ['']
feats_2 = ['']
feats_3 = ['']
feats_4 = ['']
for row in range(Cases):
    if(Phrase_label[row]==0):
        feats_0.append(Phrase[row])
    if(Phrase_label[row]==1):
        feats_1.append(Phrase[row])
    if(Phrase_label[row]==2):
        feats_2.append(Phrase[row])
    if(Phrase_label[row]==3):
        feats_3.append(Phrase[row])
    if(Phrase_label[row]==4):
        feats_4.append(Phrase[row])
feats_0 = feats_0[1:len(feats_0)]
feats_1 = feats_1[1:len(feats_1)]
feats_2 = feats_2[1:len(feats_2)]
feats_3 = feats_3[1:len(feats_3)]
feats_4 = feats_4[1:len(feats_4)]
'''
#negids = movie_reviews.fileids('neg')
#posids = movie_reviews.fileids('pos')
'''

feats0 = [(word_feats(  find_bigrams( str(feats_0[k]).split() ) ), '0') for k in range(0,len(feats_0),1)]
feats1 = [(word_feats(  find_bigrams( str(feats_1[k]).split() ) ), '1') for k in range(0,len(feats_1),1)]
feats2 = [(word_feats(  find_bigrams( str(feats_2[k]).split() ) ), '2') for k in range(0,len(feats_2),1)]
feats3 = [(word_feats(  find_bigrams( str(feats_3[k]).split() ) ), '3') for k in range(0,len(feats_3),1)]
feats4 = [(word_feats(  find_bigrams( str(feats_4[k]).split() ) ), '4') for k in range(0,len(feats_4),1)]
# Real test cases , 'pop' means nothing
#testfeats = [(word_feats( str(Phrase_test[k]).split()), 'pop') for k in range(0,len(Phrase_test),1)]

#neucutoff = len(neufeats)*4/5
cutoff0 = len(feats0)*4/5
cutoff1 = len(feats1)*4/5
cutoff2 = len(feats2)*4/5
cutoff3 = len(feats3)*4/5
cutoff4 = len(feats4)*4/5

trainfeats = feats0[:cutoff0] + feats1[:cutoff1] + feats2[:cutoff2] + feats3[:cutoff3] + feats4[:cutoff4] 
testfeats = feats0[cutoff0:] + feats1[cutoff1:] + feats2[cutoff2:] + feats3[cutoff3:] + feats4[cutoff4:]

print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
#classifier = NaiveBayesClassifier.train(trainfeats)
#classifier = nltk.classify.DecisionTreeClassifier.train(trainfeats)
classifier = SklearnClassifier(BernoulliNB()).train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
#classifier.show_most_informative_features()

results = classifier.batch_classify([fs for (fs,l) in testfeats])

count = 0
'''
with open(loc_submission, "wb") as outfile:
    outfile.write("PhraseID,Sentiment\n")
    for val in results:
      outfile.write("%s,%s\n"%(df_test['PhraseId'][count],val))
      count += 1
'''
