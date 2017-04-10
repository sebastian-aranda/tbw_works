import csv
import nltk
import pickle
import gc
import sys
import time
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import defaultdict

stop = stopwords.words('english')
puncts = ".,:;?!()[]{}~+-\"\'#$%&"
#puncts = ".,?!()[]~-\"\'&1234567890"

file_path = './amazon-fine-foods/Reviews.csv'

timer = time.time()*1000
mcorpus = dict()
mcorpus['documents']= list()  #Corpus para guardar resenias
mcorpus['tokens'] = list() #Corpus para guardar tokens
mcorpus['token_freq'] = defaultdict(int) #Diccionario para contar repeticiones de tokens
i=-1
j=0
n = 10000 #Documentos procesados
with open(file_path, 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader: 
		i += 1
		if i==0:
			continue
		elif i>n:
			break

		#Eliminando etiquetas
		text = BeautifulSoup(row[9], 'html.parser').getText()

		#Guardando documento
		mcorpus['documents'].append(text.split())
		
		#Pasando a minusculas
		text = text.lower()

		#Eliminando signos y puntuaciones
		for sym in puncts:
			text = text.replace(sym, ' ')

		#Eliminando Stopwords
		filtered = [token for token in text.split() if token not in stop]
			
		for token in filtered:
			mcorpus['tokens'].append(token)
			mcorpus['token_freq'][token] += 1

		sys.stdout.write("\rDocumentos Procesados: "+str(i)+"/"+str(n))
		sys.stdout.flush()

		# Guardando avance en pickle cada 10000 registros
		if i%10000 == 0:
			j += 1
			pickle_file_path = './corpus/corpus_'+str(j)+'.pkl'
			output = open(pickle_file_path, 'wb')
			pickle.dump(mcorpus, output)
			output.close()
			
			mcorpus['documents']= list()  #Corpus para guardar resenias
			mcorpus['tokens'] = list() #Corpus para guardar tokens
			mcorpus['token_freq'] = defaultdict(int) #Diccionario para contar repeticiones de tokens
			gc.collect()
	
	timer = time.time()*1000-timer
	print('\nTiempo de procesamiento: ' + str(timer) + ' ms')

	