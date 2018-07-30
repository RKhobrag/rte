import numpy as np
import gensim.models.keyedvectors as word2vec

class embeddings:
	embeddings_index = {}
	def __init__(self, path):	
		self.model = word2vec.KeyedVectors.load_word2vec_format(path, binary=True) 	
		# word_embeddings = open(path)
		# i=0
		# for line in word_embeddings:
		#     values = line.split()
		#     word = values[0]
		#     try:
		#     	coefs = np.asarray(values[1:], dtype='float32')
		#     	pass
		#     except :
		#     	pass
		    	
		#     self.embeddings_index[word] = coefs
		#     i+=1
		# word_embeddings.close()
		
	def embed(self, data):
		embedded = []
		for sent in data:
			s = []
			for word in sent:
				if word in self.model:
					s.append(self.model[word])
				else:
					s.append(np.zeros(300))
			embedded.append(np.array(s))
		return np.array(embedded)
