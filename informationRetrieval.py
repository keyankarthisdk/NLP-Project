from util import *

# Add your import statements here




class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docs_map = {}

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = None

		#Fill in code here
		index = {}
		for doc_i in range(len(docs)):
			self.docs_map[docIDs[doc_i]] = docs[doc_i]
			doc = docs[doc_i]
			uniqueTermsInDoc = []
			for sentence in doc:
				uniqueTermsInDoc = uniqueTermsInDoc + sentence
			uniqueTermsInDoc = list(set(uniqueTermsInDoc))
			for term in uniqueTermsInDoc:
				if term not in index.keys():
					index[term.lower()] = [docIDs[doc_i]]
				else:
					index[term.lower()].append(docIDs[doc_i])

		self.index = index
		


	def rank(self, queries, **params):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
			[[['s1'],['s2']]]
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		for i in range(len(queries)):
			query = queries[i]
			terms = []
			merged_sentences = ""
			for sentence in query:
				merged_sentences = merged_sentences + " " + " ".join(sentence)
				terms.extend(sentence)
			terms = list(set(terms))

			# Get Vectorizer and Rank
			docs = []
			doc_IDs = []
			doc_id_map = {}
			for term in terms:
				term = term.lower()
				if term in self.index.keys():
					for d in self.index[term]:
						doc_id_map[d] = self.docs_map[d]
			for k in doc_id_map.keys():
				merged_doc = ""
				for sentence in doc_id_map[k]:
					merged_doc = merged_doc + " " + " ".join(sentence)
				docs.append(merged_doc)
				doc_IDs.append(k)
			if len(list(doc_id_map.keys())) >= 1:
				vectorizer, doc_vector_matrix = Vectorise_Docs(docs)
				query_vect_matrix = Vectorise_Query(vectorizer, merged_sentences)

				query_vect_matrix = query_vect_matrix.toarray()
				doc_vector_matrix = doc_vector_matrix.toarray()

				# Apply Weights
				feature_names = vectorizer.get_feature_names()
				weights = params["sim_weights"][i]
				weightsArray = []
				for f in feature_names: weightsArray.append(weights[f] if f in weights.keys() else 1.0)
				weightsArray = np.array(weightsArray)
				# print(feature_names[np.logical_and(weightsArray == 0.0, query_vect_matrix[0] != 0.0)[0]])
				# print(query_vect_matrix.shape, weightsArray.shape, len(feature_names))
				query_vect_matrix2 = np.multiply(query_vect_matrix[0], weightsArray)
				# print(np.sum(query_vect_matrix2 - query_vect_matrix[0]))
				query_vect_matrix = query_vect_matrix2.reshape(1, -1)
				# print(query_vect_matrix.shape)

				cosine_similarities = GetSimilarity(query_vect_matrix, doc_vector_matrix)
				query_rank = [x for _, x in sorted(zip(cosine_similarities, doc_IDs), reverse=True)]
				doc_IDs_ordered.append(query_rank)
			else:
				doc_IDs_ordered.append([])
	
		return doc_IDs_ordered


	def doc_term_mat(data):
		"""
		Generating term dictinoary of the dataset and document term matrix

		Parameters
		----------
		arg1 : 
		

		Returns
		-------
		Term dictionary and document term matrix
		"""
		# generate term dictionary
		_dict = corpora.Dictionary(data)
		# convert tokenized documents into a document corpus
		doc_term_matrix = [dictionary.doc2bow(doc) for doc in data]

		return _dict, doc_term_matrix



	def LSA(doc_term_matrix):
		"""
		Applying LSA on the document term matrix

		Parameters
		----------
		arg1 : 
		

		Returns
		-------
		LSA model

		"""
		# generate LSA model
		lsa_model = LsiModel(doc_term_matrix, num_topics=2, id2word=dictionary)
		return lsa_model