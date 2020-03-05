from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

class TextFilterComposite:
    def __init__ (self, filters):
        self._filters = filters

    def _filter (self, tokens):
        result = tokens
        for f in self._filters:
            result = f.filter(result)
        return (' ').join(result)

    def execute (self, texts_list):
        print('===== Executing Text Filter =====')
        result = []
        for text in texts_list:
            tokens = word_tokenize(text['content'])
            filtered_text = self._filter(tokens)
            result.append({
                'content': filtered_text.lower(),
                'category': text['category']
            })

        return result

class LemmatizerFilter:
    def __init__ (self):
        print('===== Configure the lemmatizer =====')
        self._lemmatizer = WordNetLemmatizer()

    def filter (self, tokens):
        tags = pos_tag(tokens)
        return [ self._lemmatizer.lemmatize(token[0], pos=token[1][0].lower())
                    if token[1][0].lower() in ('a', 'n', 'v', 'r')
                    else self._lemmatizer.lemmatize(token[0])
                    for token in tags ]


class StopWordsFilter:
    def __init__ (self):
        print('===== Configuring stop words removal =====')

    def filter (self, tokens):
        return [ word for word in tokens
                 if not word.lower() in stopwords.words('english') ]

class PorterStemmerFilter:
    def __init__ (self):
        print('===== Configuring words stemming =====')
        self._stemmer = PorterStemmer()

    def filter (self, tokens):
        return [ self._stemmer.stem(word.lower()) for word in tokens ]
