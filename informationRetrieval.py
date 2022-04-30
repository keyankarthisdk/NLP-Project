from calendar import c
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
		for qi in range(len(queries)):
			query = queries[qi]
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
			# Inverted Index Reduce Doc Set
			if params["inv_index_reduce"]:
				for term in terms:
					term = term.lower()
					if term in self.index.keys():
						for d in self.index[term]:
							doc_id_map[d] = self.docs_map[d]
			else:
				doc_id_map = dict(self.docs_map)
			# Merge Documents
			for k in doc_id_map.keys():
				merged_doc = ""
				for sentence in doc_id_map[k]:
					merged_doc = merged_doc + " " + " ".join(sentence)
				docs.append(merged_doc)
				doc_IDs.append(k)

			# Construct Vectors and Rank
			if len(list(doc_id_map.keys())) >= 1:
				# Clean Docs and Query Words if using Word2Vec Model
				if params["model_clean_text"]:
					docs = [Word2Vec_CleanText(params["model_word2vec"], doc) for doc in docs]
					merged_sentences = Word2Vec_CleanText(params["model_word2vec"], merged_sentences)

				# Get TFIDFs and Feature Name Maps
				vectorizer, docs_tfidf_matrix = Vectorise_Docs_TFIDF(docs)
				query_tfidf_vector = Vectorise_Query_TFIDF(vectorizer, merged_sentences)
				query_tfidf_vector = query_tfidf_vector.toarray()
				docs_tfidf_matrix = docs_tfidf_matrix.toarray()
				feature_names = vectorizer.get_feature_names_out()
				feature_names_map = {feature_names[i]: i for i in range(len(feature_names))}

				# Apply Query Expansion Weights
				weights = params["sim_weights"][qi]
				weightsArray = []
				for f in feature_names: weightsArray.append(weights[f] if f in weights.keys() else 1.0)
				weightsArray = np.array(weightsArray)
				query_tfidf_eff_vector = np.multiply(query_tfidf_vector[0], weightsArray)
				query_tfidf_vector = query_tfidf_eff_vector
				
				# Get Query and Doc Vectors
				if params["vector_type"] == "TFIDF Stacking":
					docs_final_matrix = np.array(docs_tfidf_matrix)
					query_final_vector = np.reshape(query_tfidf_vector, (1, -1))
				elif params["vector_type"] == "Word2Vec Without TFIDF":
					# Get Doc Word2Vec Vectors
					docs_final_matrix = []
					for doc in docs:
						doc_words_vectors = np.array([
							Word2Vec_GetWordVector(params["model_word2vec"], word) for word in doc.split()
						])
						doc_final_vector = np.mean(doc_words_vectors, axis=0)
						docs_final_matrix.append(doc_final_vector)
					docs_final_matrix = np.array(docs_final_matrix)
					# Get Query Word2Vec Vector
					query_words_vector = np.array([
						Word2Vec_GetWordVector(params["model_word2vec"], word) for word in merged_sentences.split()
					])
					query_final_vector = np.mean(query_words_vector, axis=0)
					query_final_vector = np.reshape(query_final_vector, (1, -1))
				elif params["vector_type"] == "Word2Vec With TFIDF":
					# Get Doc Word2Vec Vectors
					docs_final_matrix = []
					for i in range(len(docs)):
						doc = docs[i]
						doc_tfidf_vector = docs_tfidf_matrix[i]
						doc_words = list(set(doc.split()))
						words_weightages = [
								doc_tfidf_vector[feature_names_map[word]] if word in feature_names_map.keys() 
								else doc_tfidf_vector.min() 
							for word in doc_words
						]
						doc_words_vectors = np.array([
							Word2Vec_GetWordVector(params["model_word2vec"], doc_words[j]) * words_weightages[j] 
							for j in range(len(doc_words))
						])
						doc_final_vector = np.mean(doc_words_vectors, axis=0)
						docs_final_matrix.append(doc_final_vector)
					docs_final_matrix = np.array(docs_final_matrix)
					# Get Query Word2Vec Vector
					query_words = list(set(merged_sentences.split()))
					words_weightages = [
							query_tfidf_vector[feature_names_map[word]] if word in feature_names_map.keys() 
							else query_tfidf_vector.min() 
						for word in query_words
					]
					query_words_vector = np.array([
						Word2Vec_GetWordVector(params["model_word2vec"], query_words[j]) * words_weightages[j] 
						for j in range(len(query_words))
					])
					query_final_vector = np.mean(query_words_vector, axis=0)
					query_final_vector = np.reshape(query_final_vector, (1, -1))

				# print("Query Vector:", query_final_vector.shape)
				# print("Doc Matrix:", docs_final_matrix.shape)

				# Get Similarity
				cosine_similarities = GetSimilarity(query_final_vector, docs_final_matrix)
				query_rank = [x for _, x in sorted(zip(cosine_similarities, doc_IDs), reverse=True)]
				doc_IDs_ordered.append(query_rank)
			else:
				doc_IDs_ordered.append([])
			
			# Update Progress
			print(qi, "of", len(queries))
			params["progress_obj"]("Ranking: ", qi / len(queries))
	
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