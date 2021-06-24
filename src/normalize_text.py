from bs4 import BeautifulSoup
import re
import unicodedata
import contractions
from word2number import w2n
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from emoji import demojize
from gingerit.gingerit import GingerIt
from googletrans import Translator

def remove_HTML_tags(text):
	'''
	'&amp;' is converted to 
	'&'
	'''
	return BeautifulSoup(text, "html.parser").get_text()

def convert_accented_chars(text):
	'''
	'Café' is converted to 
	'Cafe'
	'''
	return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def expand_contracted_words(text):
	'''
	'I'd' is converted to 
	'I would'
	'''

	ret = []    
	for word in text.split():
		ret.append(contractions.fix(word))
	return ' '.join(ret)

def remove_names_all(text):

	'''
	For example: @PMOIndia, @musicaltrees is removed.
	'''

	ret = []
	for word in text.split():
		if word[0] != '@':
			ret.append(word)
	return ' '.join(ret)

def remove_special_chars(text):

	'''
	'Solitude is not the absence of Love❤️ 123' is converted to
	'Solitude is not the absence of Love red_heart'.

	Note: #s are not removed due to the dataset having tweets.
	'''

	pattern = r'[^#a-zA-z\s]'

	text = demojize(text)
	text = ' '.join(text.split(':'))

	return re.sub(pattern, '', text)

def lowercase_text_all(text):

	'''
	'HeLlO' is converted to 
	'hello'
	'''

	return text.lower()

def numberwords_to_numeric(text):

	'''
	'I am five years old' is converted to
	'I am 5 years old'
	'''

	text = word_tokenize(text)

	ret = []
	for word in text:
		if nltk.pos_tag([word]) == 'CD':
			ret.append(w2n.word_to_num(word))
		else:
			ret.append(word)
	return ' '.join(ret)

def remove_stopwords_all(text):

	'''
	'I am feeling good.' is converted to
	'feeling good .'
	'''

	stop_words = stopwords.words('english')

	text = word_tokenize(text)
	ret = []
	for word in text:
		if word.lower() not in stop_words:
			ret.append(word)
	return ' '.join(ret)

def remove_links_all(text):
    
    return ' '.join([re.sub(r'(https?://\S+)','',word)for word in text.split()])

def spellcheck(text):

	parser = GingerIt()
	text = parser.parse(text)['result']

	return text

def translate_all(text):

	translator = Translator()
	return translator.translate(text).text

def normalize(text, remove_HTML = True, convert_accented = True, expand_contractions = True,
	remove_special = True, lowercase_text = True, numberwords_numeric = True, remove_stopwords = True, 
	remove_names = True, remove_links = True, correct_spelling = False, translate = False):

	if lowercase_text:
		text = lowercase_text_all(text)

	if remove_HTML:
		text = remove_HTML_tags(text)

	if remove_names:
		text = remove_names_all(text)

	if remove_links:
		text = remove_links_all(text)

	tmp = ""
	for j in range(len(text)):
		if text[j] in ',.':
			tmp += text[j] + ' '
		else:
			tmp += text[j]

	text = tmp

	if remove_special:
		text = remove_special_chars(text)

	if convert_accented:
		text = convert_accented_chars(text)

	if expand_contractions:
		text = expand_contracted_words(text)

	if numberwords_numeric:
		text = numberwords_to_numeric(text)

	if remove_stopwords:
		text = remove_stopwords_all(text)

	if translate:
		text = translate_all(text)

	if correct_spelling:
		text = spellcheck(text)

	text = word_tokenize(text)
	lemmatizer = WordNetLemmatizer()

	text = [lemmatizer.lemmatize(word) for word in text]
	op = []
	j = 0
	while j < len(text):
		if text[j] == '#' and j < len(text) - 1:
			op.append(text[j] + text[j+1])
			j += 2
		elif text[j] == '#' and j == len(text) - 1:
			j += 1
		else:
			op.append(text[j])
			j += 1
	text = ' '.join(op)

	return text