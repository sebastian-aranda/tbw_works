import pickle
import gc
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

log = {'corpus_read_time': 0, 'filtering_infrequent_tokens_time': 0, 'top30bigrams_time': 0, 'postagger_time': 0, 'nertagger_time': 0}

total_timer = time.time()*1000

#Leyendo corpus desde archivo pickle (10.000 registros) creado previamente
print("Leyendo Corpus...")
corpus_timer = time.time()*1000
bigcorpus = dict()
bigcorpus['documents'] = list()
for i in range(1):
	i += 1
	pkl_file = open('./corpus/corpus_'+str(i)+'.pkl', 'rb')
	mcorpus = pickle.load(pkl_file) 
	pkl_file.close()
	bigcorpus['documents'] += mcorpus['documents']

corpus_timer = time.time()*1000 - corpus_timer
log['corpus_read_time'] = corpus_timer
print('\nTiempo de procesamiento: ' + str(corpus_timer) + ' ms')

#Entrena el sentiment_analyzer para ser usado en el corpus de resenias
training_timer = time.time()*1000
print("Entrenando sentiment_analyzer...")
n_instances = 1000
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
train_subj_docs = subj_docs[:750]
test_subj_docs = subj_docs[750:1000]
train_obj_docs = obj_docs[:750]
test_obj_docs = obj_docs[750:1000]
sentiment_analyzer = SentimentAnalyzer()
training_docs = train_subj_docs+train_obj_docs
test_docs = test_subj_docs+test_obj_docs
all_words_neg = sentiment_analyzer.all_words([mark_negation(doc) for doc in training_docs])
unigram_feats = sentiment_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentiment_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
training_set = sentiment_analyzer.apply_features(training_docs)
test_set = sentiment_analyzer.apply_features(test_docs)
trainer = NaiveBayesClassifier.train
classifier = sentiment_analyzer.train(trainer, training_set)
for key,value in sorted(sentiment_analyzer.evaluate(test_set).items()):
	print('{0}: {1}'.format(key, value))
training_timer = time.time()*1000 - training_timer
print("Tiempo de procesamiento: " + str(training_timer) + ' ms')
#Se evalua la polaridad de las resenias cargadas en bigcorpus['documents']
polarity_timer = time.time()*1000
print("Evaluando...")
pickle_file_path = './polarity/polarity.pkl'
output = open(pickle_file_path, 'wb')
stop = stopwords.words('english')
polarity = list()
sid = SentimentIntensityAnalyzer()
for sent in bigcorpus['documents']:
        lower = [token.lower() for token in sent]
        sentence_filtered = [token for token in lower if token not in stop]
        ss = sid.polarity_scores(" ".join(sentence_filtered))
        polarity.append(ss)
        # Descomentar para tener resultados por pantalla
##        for k in sorted(ss):
##                print('{0} : {1}'.format(k, ss[k]))
pickle.dump(polarity, output)
output.close()
gc.collect()
print("Terminado")
polarity_timer = time.time()*1000 - polarity_timer
print("Tiempo de procesamiento: " + str(polarity_timer) + ' ms')
