from util import *

# Add your import statements here




class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = None

		#Fill in code here
		ignoreTokens = set(stopwords.words('english')).union(set(string.punctuation)).union(set(string.octdigits))
		stopwordRemovedText = [[token for token in sentence if not token.lower() in ignoreTokens] for sentence in text]

		return stopwordRemovedText