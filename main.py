# Imports
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from util import *

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

# Main Vars
TITLE_WEIGHTAGE = 1
PRINT_OBJ = print
PROGRESS_OBJ = None

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

		self.informationRetriever = InformationRetrieval()
		self.evaluator = Evaluation()


	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)
		elif self.args.tokenizer == "ngram":
			return self.tokenizer.ngram_tokenizer(text, self.args.params["ngram_n"])

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)


	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""
		global PROGRESS_OBJ

		# Segment queries
		segmentedQueries = []
		i = 0
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			# Spell Correction
			segmentedQuery = [SpellCorrect(sentence) for sentence in segmentedQuery]
			segmentedQueries.append(segmentedQuery)
			i += 1
			if PROGRESS_OBJ is not None: PROGRESS_OBJ("Query: Sentence Segmentation", i / len(queries))
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'), indent=4)
		# Tokenize queries
		tokenizedQueries = []
		i = 0
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
			i += 1
			if PROGRESS_OBJ is not None: PROGRESS_OBJ("Query: Tokenization", i / len(segmentedQueries))
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'), indent=4)
		# Stem/Lemmatize queries
		reducedQueries = []
		i = 0
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
			i += 1
			if PROGRESS_OBJ is not None: PROGRESS_OBJ("Query: Inflection Reduction", i / len(tokenizedQueries))
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'), indent=4)
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		i = 0
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
			i += 1
			if PROGRESS_OBJ is not None: PROGRESS_OBJ("Query: Stopword Removal", i / len(reducedQueries))
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'), indent=4)

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		global PROGRESS_OBJ
		
		# Segment docs
		segmentedDocs = []
		i = 0
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
			i += 1
			if PROGRESS_OBJ is not None: PROGRESS_OBJ("Doc: Sentence Segmentation", i / len(docs))
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'), indent=4)
		# Tokenize docs
		tokenizedDocs = []
		i = 0
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
			i += 1
			if PROGRESS_OBJ is not None: PROGRESS_OBJ("Doc: Tokenization", i / len(segmentedDocs))
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'), indent=4)
		# Stem/Lemmatize docs
		reducedDocs = []
		i = 0
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
			i += 1
			if PROGRESS_OBJ is not None: PROGRESS_OBJ("Doc: Inflection Reduction", i / len(tokenizedDocs))
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'), indent=4)
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		i = 0
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
			i += 1
			if PROGRESS_OBJ is not None: PROGRESS_OBJ("Doc: Stopword Removal", i / len(reducedDocs))
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'), indent=4)

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs


	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
		queries_json = json.load(open(self.args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids = [item["id"] for item in docs_json]
		docs = [item["body"] for item in docs_json]
		docTitles = [item["title"] for item in docs_json]
		# Include Titles
		docs = [IncludeTitleInDoc(doc, title, TITLE_WEIGHTAGE) for doc, title in zip(docs, docTitles)]

		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for each query
		doc_IDs_ordered = self.informationRetriever.rank(processedQueries)

		# Read relevance judements
		qrels = json.load(open(self.args.dataset + "cran_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			PRINT_OBJ("Precision, Recall and F-score @ " +  
				str(k) + " : " + str(precision) + ", " + str(recall) + 
				", " + str(fscore))
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)
			PRINT_OBJ("MAP, nDCG @ " +  
				str(k) + " : " + str(MAP) + ", " + str(nDCG))

		# Plot the metrics and save plot 
		plt.plot(range(1, 11), precisions, label="Precision")
		plt.plot(range(1, 11), recalls, label="Recall")
		plt.plot(range(1, 11), fscores, label="F-Score")
		plt.plot(range(1, 11), MAPs, label="MAP")
		plt.plot(range(1, 11), nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		plt.savefig(self.args.out_folder + "eval_plot.png")

		
	def handleCustomQuery(self, query=None):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		#Get query
		if query is None:
			print("Enter query below")
			query = input()
		# Process documents
		processedQuery = self.preprocessQueries([query])[0]

		# Read documents
		docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids = [item["id"] for item in docs_json]
		docs = [item["body"] for item in docs_json]
		docTitles = [item["title"] for item in docs_json]
		# Include Titles
		docs = [IncludeTitleInDoc(doc, title, TITLE_WEIGHTAGE) for doc, title in zip(docs, docTitles)]

		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for the query
		doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

		# Print the IDs of first five documents
		PRINT_OBJ("\nTop five document IDs : ")
		for id_ in doc_IDs_ordered[:5]:
			PRINT_OBJ(id_)



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
