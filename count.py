import nltk.classify.util
import pandas as pd
import numpy as np
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import csv
import re
import random
 
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
words = {}
Phrase_label = ['']
data = open('train.csv','r')
counts = [0,0,0,0,0]
pattern = re.compile("[ \&\,\.\/\(\)\-\_\:\~\<\>\+\'\!\t\?\[\]]+")
p_dict = {}
for row in csv.DictReader(data):
	phrase = row['Phrase'].lower()
	label = row['Sentiment']
# 	ph = re.sub(pattern, ' ', phrase).lstrip(' ').rstrip(' ')
	ph = phrase
	if len(ph.split(' ')) == 1 and ph != '':
		words[ph] = int(label)
    	counts[int(label)] += 1
	Phrase.append(phrase)
	p_dict[phrase] = label
	Phrase_label.append(label)
data.close()

Phrase = Phrase[1:len(Phrase)]
Phrase_label = Phrase_label[1:len(Phrase_label)]
Phrase_label = [int(numeric_string) for numeric_string in Phrase_label]
percentage_feats = []
for i in range(len(Phrase)):
	phrase = Phrase[i]
	label = Phrase_label[i]
# 	ph = re.sub(pattern, ' ', phrase).lstrip(' ').rstrip(' ')
	ph = phrase
	sp = ph.split(' ')
	ph_len = len(sp)
	sens = [0, 0, ph_len, 0, 0]
	not_flag = False
	if ph_len > 1:
		for k in range(ph_len):
			w = sp[k]
			if w in words:
				if w == 'not' or w == 'n\'t' or w == 'no' or w == 'never' or w == 'none' or w == 'nothing' or w == 'negative':
					not_flag = True
				elif w == '.' or w == ',' or w == ';':
					not_flag = False
				elif w== 'however' or w == 'but' or w=='instead' or w=='nevertheless' or w=='while':
					if k < ph_len - 1:
						not_flag = False
						sens = [0,0,ph_len-k-1,0,0]
			#	elif w=='although' or w=='despite':
				else:
					if not_flag:
						if words[w] > 2:
							sens[1] += 1
							sens[2] -= 1
						elif words[w] < 2:
							sens[3] += 1
							sens[2] -= 1
					else:
						sens[words[w]] += 1
						sens[2] -= 1
		p_feat = {}
		ph_len = ph_len - sens[2] + 1
		sens[2] = 0
		for j in range(5):
			p_feat[str(j)] = str(round((float(sens[j])*10.0/float(ph_len))))
		percentage_feats.append((p_feat, str(label)))

print counts


Cases = len(Phrase)


cutoff = len(percentage_feats)*4/5

trainfeats = percentage_feats
#testfeats = percentage_feats[cutoff:] #+ neufeats[neucutoff:Cases]

#print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)
precision = 0
for feat in trainfeats:
	pred = int(classifier.classify(feat[0]))
#	if pred != 0 and pred != 4:
#		ran = random.randint(0,20)
#		if ran == 0:
#			pred -= 1
#		elif ran == 1:
#			pred += 1
	if pred == int(feat[1]):
		precision += 1
print 'accuracy:', float(precision) / float(len(trainfeats))
#print 'accuracy:', nltk.classify.util.accuracy(classifier, trainfeats)
classifier.show_most_informative_features()
#raw_input()

outfile = open('loc_submission.csv', 'w')
outfile.write("PhraseID,Sentiment\n")
data = open('test.csv','r')
counts = [0,0,0,0,0]
for row in csv.DictReader(data):
	phrase = row['Phrase'].lower()
	pid = row['PhraseId']
 	ph = re.sub(pattern, ' ', phrase).lstrip(' ').rstrip(' ')
	sp = ph.split(' ')
	ph_len = len(sp)
	sens = [0, 0, ph_len, 0, 0]
	label = 2
	if ph_len > 1:
		if ph in p_dict:
			outfile.write('%s,%s\n'%(pid, str(p_dict[ph])))
		else:
			not_flag = False
			for k in range(ph_len):
				w = sp[k]
				if w in words:
					if w == 'not' or w == 'n\'t' or w == 'no' or w == 'never' or w == 'none' or w == 'nothing' or w == 'negative':
						not_flag = True
					elif w == '.' or w == ',' or w == ';':
						not_flag = False
					elif w== 'however' or w == 'but' or w=='instead' or w=='nevertheless' or w=='while':
						if k < ph_len - 1:
							not_flag = False
							sens = [0,0,ph_len-k-1,0,0]
					else:
						if not_flag:
							if words[w] > 2:
								sens[1] += 1
								sens[2] -= 1
							elif words[w] < 2:
								sens[3] += 1
								sens[2] -= 1
						else:
							sens[words[w]] += 1
							sens[2] -= 1
			p_feat = {}
			ph_len = ph_len - sens[2] + 1
			sens[2] = 0
			for j in range(5):
				p_feat[str(j)] = str(round((float(sens[j])*10.0/float(ph_len))))
			label = int(classifier.classify(p_feat))
			outfile.write('%s,%s\n'%(pid, label))
	else:
		if ph in words:
			label = words[ph]
			outfile.write('%s,%s\n'%(pid, str(words[ph])))
		else:
			label = 2
			outfile.write('%s,2\n'%(pid))
	counts[label] += 1
			
data.close()
outfile.close()
print counts
#Phrase_test = Phrase_test[1:len(Phrase_test)]
''' 
df_train = pd.DataFrame(pd.read_csv(loc_train))
'''
#df_test = pd.DataFrame(pd.read_csv(loc_test))

#count = 0

#with open(loc_submission, "wb") as outfile:
#    outfile.write("PhraseID,Sentiment\n")
#    for val in results:
#    outfile.write("%s,%s\n"%(df_test['PhraseId'][count],val))
#      count += 1

