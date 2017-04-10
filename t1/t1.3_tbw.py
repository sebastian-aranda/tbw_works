import nltk
import pickle
import time

from_index = 0
to_index = 4

#Leyendo Documentos Postagged
for i in range(from_index, to_index):
	pkl_file = open('./postags/postagged_'+str(i+1)+'.pkl', 'rb')
	postags = pickle.load(pkl_file) 
	pkl_file.close()
	print('Postags Documento '+str(i+1)+'\n'+str(postags))


print('\n')

#Leyendo Documentos Nertagged
for i in range(from_index, to_index):
	pkl_file = open('./nertags/nertagged_'+str(i+1)+'.pkl', 'rb')
	nertags = pickle.load(pkl_file) 
	pkl_file.close()
	print('Nertags Documento '+str(i+1)+'\n'+str(nertags))