from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string


def get_tokens(input_text):
    """
    A tokenizer that splits a string using a regular expression, 
    which matches either the tokens or the separators between tokens 

    Parameters
    ----------
    text :  input text  

    Returns
    -------        
    tokens
    """
    return RegexpTokenizer('\w+|\$[\d\.]+|\S+').tokenize(input_text)


def lemmatize(input_text):
    """
    Lemmatize using WordNet's built-in morphy function.
    Returns the input word unchanged if it cannot be found in WordNet.

    Parameters
    ----------
    text :  input text  

    Returns
    -------        
    tokens
    """
    for i in range(len(input_text)):
        input_text[i] = WordNetLemmatizer().lemmatize(input_text[i])
    return input_text


def remove_stopwords(test_point):
    """
    Remove stopwords from the input text
    -------
    Returns
    -------        
    tokens without the stopwords
    """
    neutral_words = ['could', 'might', 'would', 'may', 'shall',
                     'www', 'http', 'email', 'sent', 'send', 'subject']
    special_characters = ['+', '-', '_', '?', '<=', '>=', '>', '<',
                          '(', ')', '{', '}',  '[', ']', '"', ';', ':', '!', '*', '@', '#', '$', '%', '&', '~', ',', '.', '\ ',  '/']

    updated_stop_words = special_characters + \
        neutral_words + list(stopwords.words('english'))
    modified_test_point = []
    for word in test_point:
        if word not in updated_stop_words and len(word) > 2:
            modified_test_point.append(
                word.translate(string.punctuation).lower())
    return modified_test_point


def remove_mail_id(email):
    modified_email = []
    for m in email:
        if (('@' not in m) or ('.' not in m)):
            modified_email.append(m)
    return modified_email

# Removes formatting


def remove_formatting(test_point):
    format_words = ['\\', '{', '}', '.', ',', ';', ':']
    modified_test_point = []
    for word in test_point:
        if word[0] not in format_words:
            modified_test_point.append(word)
    return modified_test_point

# Removes numbers and punctutations


def remove_numbers_punctuations(test_point):
    punctuations = list(string.punctuation)
    modified_test_point = []
    for word in test_point:
        modified_test_point.append(
            ''.join([i for i in word if not i.isdigit() and i not in punctuations]))
    return modified_test_point
