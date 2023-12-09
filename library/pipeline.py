import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re

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
            self.stop_words = set(['i', 'feel', 'and', 'to', 'the', 'a', 'feeling', 'that', 'of', 'my', 'in', 'it', 'like', 'so', 'for', 'im', 'me', 'but', 'was', 'have'])
        except Exception as e:
            print("[ERR] The following error occured while trying to Pre-Process the text: "+str(e))

    def normalize_text(self, text:str) -> str:
        """
        This method normalizes the given text:
        - Lowers entire text into lowercase for case consistency.
        - Replaces words like didn't with didnt.
        - Extracts only textual information discarding the rest.
        """
        try:
            text = text.lower()
            text = text.replace("'","")
            text = re.findall("[a-z]+", text)
            text = " ".join(text)
            return text
        except Exception as e:
            print("[ERR] The following error occured while trying to normalize the text: "+str(e))

    def remove_most_frequent_stopword(self, text:str) -> str:
        """
        This method attempts to remove the most frequent stop words within a given sentence.
        """
        try:
            # Tokenize the text
            words = word_tokenize(text)

            filtered_text = ' '.join([word for word in words if word.lower() not in self.stop_words])

            return filtered_text
        except Exception as e:
            print("[ERR] The following error occured while trying to removing the most frequent stop words from the string: "+str(e))

    def transform_text(self, text:str) -> str:
        """
        Implements both normalize and stopwords method to get the target string.
        """
        text = self.normalize_text(text= text)
        text = self.remove_most_frequent_stopword(text= text)
        return text
    
