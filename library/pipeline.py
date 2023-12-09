import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

class Pipeline():
    """
    This Custom module aims to preprocess user input text from end to end.
    - It normalizes the input text:
        - Lowers entire text into lowercase for case consistency.
        - Extracts only textual information discarding the rest.
        - Removes stop words.
    - And then generates word embeddings.
    """
    stop_words:list = None 

    def __init__(self) -> None:
        try:
            nltk.download('stopwords')
            nltk.download('punkt')

            self.stop_words = set(stopwords.words("english"))
        except Exception as e:
            print("[ERR] The following error occured while trying to Pre-Process the text: "+str(e))