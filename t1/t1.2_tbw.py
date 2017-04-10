import nltk
import pickle
import gc
import time
import sys
from nltk.collocations import *
from collections import defaultdict
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger


#Funcion que elimina token con repeticiones menores a x
def dict_tokens_modified(data, x):
    new_dict_tokens = {k: v for k, v in data.iteritems() if v >= x}
    return new_dict_tokens

log = {'corpus_read_time': 0, 'filtering_infrequent_tokens_time': 0, 'top30bigrams_time': 0, 'postagger_time': 0, 'nertagger_time': 0}

total_timer = time.time()*1000

#Leyendo corpus desde archivo pickle (10.000 registros) creado previamente
print("Leyendo Corpus...")
corpus_timer = time.time()*1000
bigcorpus = dict()
bigcorpus['documents']= list()  #Corpus para guardar resenias
bigcorpus['tokens'] = list() #Corpus para guardar tokens
bigcorpus['token_freq'] = defaultdict(int) #Diccionario para contar repeticiones de tokens
tokens_per_document = list()
for i in range(1):
	i += 1
	pkl_file = open('./corpus/corpus_'+str(i)+'.pkl', 'rb')
	mcorpus = pickle.load(pkl_file) 
	pkl_file.close()

	bigcorpus['documents'] += mcorpus['documents'][:100]
	for doc in mcorpus['documents']:
		tokens_per_document.append(len(doc))
	bigcorpus['tokens'] += mcorpus['tokens']
	bigcorpus['token_freq'] = {k: bigcorpus['token_freq'].get(k,0)+mcorpus['token_freq'].get(k,0) for k in set(bigcorpus['token_freq']) | set(mcorpus['token_freq'])}

corpus_timer = time.time()*1000 - corpus_timer
log['corpus_read_time'] = corpus_timer

print("Largo Conjunto de Tokens del Corpus: "+str(len(bigcorpus['tokens'])))
print("Tokens por Documento Promedio: "+str(sum(tokens_per_document)/len(tokens_per_document)))
print("Tokens por Documento Maximo: "+str(max(tokens_per_document)))
print("Tokens por Documento Minimo: "+str(min(tokens_per_document)))

#Eliminando palabras con frecuencia menor a 3
print("Filtrando palabras con frecuencia menor a 3...")
filter_infreq_timer = time.time()*1000
filtered_dict = dict_tokens_modified(bigcorpus['token_freq'], 3)
validated_tokens = list(filtered_dict)
bigcorpus['tokens'] = [token for token in bigcorpus['tokens'] if token in validated_tokens]
bigcorpus['token_freq'] = filtered_dict

filter_infreq_timer = time.time()*1000-filter_infreq_timer
log['filtering_infrequent_tokens_time'] = filter_infreq_timer

#Determinando Top30 Collocations
print("Determinando Top30 Collocations...")
bigram_timer = time.time()*1000
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(bigcorpus['tokens'])
finder.apply_freq_filter(3)
bigrams = finder.nbest(bigram_measures.pmi, 30)

bigram_timer = time.time()*1000 - bigram_timer
log['top30bigrams_time'] = bigram_timer

print('\nTOP 30 Collocations Amazon Fine Foods Reviews\n')
i = 0
for bigram in bigrams:
	i += 1
	print str(i)+'. '+str(bigram)

print('\n')
#Ejecutando Postagger y Nertagger con n documentos
postagger = StanfordPOSTagger('./stanford-postagger/models/english-bidirectional-distsim.tagger', './stanford-postagger/stanford-postagger.jar', encoding='utf-8')
nertagger = StanfordNERTagger('./stanford-nertagger/classifiers/english.all.3class.distsim.crf.ser.gz', './stanford-nertagger/stanford-ner.jar', encoding='utf-8')
postagger_times = list()
nertagger_times = list()
i = 0
n = 100 #Cantidad de documentos a taggear
print("Ejecutando Postagger y Nertagger...")
for doc in bigcorpus['documents'][i:n]:
	i += 1
	pickle_file_path_postag = './postags/postagged_'+str(i)+'.pkl'
	postagger_time = time.time()*1000
	postagged = postagger.tag(doc)
	postagger_time = time.time()*1000 - postagger_time
	postagger_times.append(postagger_time)
	output = open(pickle_file_path_postag, 'wb')
	pickle.dump(postagged, output)
	output.close()

	pickle_file_path_nertag = './nertags/nertagged_'+str(i)+'.pkl'
	nertagger_time = time.time()*1000
	nertagged = nertagger.tag(doc)
	nertagger_time = time.time()*1000 - nertagger_time
	nertagger_times.append(nertagger_time)
	output = open(pickle_file_path_nertag, 'wb')
	pickle.dump(nertagged, output)

	postagged = None
	nertagged = None
	gc.collect()

	sys.stdout.write("\rDocumentos Procesados: "+str(i)+"/"+str(n))
	sys.stdout.flush()

log['postagger_time'] = sum(postagger_times)/len(postagger_times)
log['nertagger_time'] = sum(nertagger_times)/len(nertagger_times)

pickle_file_path = './log.pkl'
output = open(pickle_file_path, 'wb')
pickle.dump(log, output)
output.close()
print('\n')

print(log)
total_timer = time.time()*1000 - total_timer
print('Largo corpus documentos: '+str(len(bigcorpus['documents']))+ ' Largo corpus tokens:'+str(len(bigcorpus['tokens'])))
print("Tiempo de procesamiento Total: "+str(total_timer) + " ms")


