from util import *

# Add your import statements here




class StopwordRemoval():

	def fromList(self, text, **params):
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

		stopwordRemovedText = text

		#Fill in code here  boundarylayercontrol
		if params["stopword_removal"]:
			stopwordRemovedText = text
			# Remove Non alphabet charecters
			# alphaRegex = re.compile("[^a-zA-Z]")
			# stopwordRemovedText = [[alphaRegex.sub(" ", token).lower().strip() for token in sentence] for sentence in text]
			# stopwordRemovedTextOld = stopwordRemovedText
			# stopwordRemovedText = []
			# for sentence in stopwordRemovedTextOld:
			# 	sentenceRemoved = []
			# 	for token in sentence:
			# 		sentenceRemoved.extend(token.split())
			# 	stopwordRemovedText.append(sentenceRemoved)
			# stopwordRemovedText = [[token for token in sentence if not token == ""] for sentence in stopwordRemovedText]
			# Remove Stop Word Tokens
			ignoreTokens = set(stopwords.words('english')).union(set(string.punctuation)).union(set(string.octdigits))
			stopwordRemovedText = [[token for token in sentence if not token.lower().strip() in ignoreTokens] for sentence in stopwordRemovedText]

		return stopwordRemovedText