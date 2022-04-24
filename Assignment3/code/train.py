from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

def pre_processing(content):
    """
    pre processing : get the tokens, remove the mail id, remove the formatter, remove the numbers and punctions , remove stopwords
    and lemmatization
    
    """
    #Step 1 :Get the tokens     
    tokens= RegexpTokenizer('\w+|\$[\d\.]+|\S+').tokenize(content)

    #Step 2 : remove mail id
    modified_tokens = []
    for m in tokens:
        if (('@' not in m) or ('.' not in m)):
            modified_tokens.append(m)
    
    #Step 3 : remove_formatting 
    format_words = ['\\', '{', '}', '.', ',', ';', ':']
    modified_test_point = []
    for word in modified_tokens:
        if word[0] not in format_words:
            modified_test_point.append(word)

    #Step 4 : remove numbers and punctuations
    punctuations = list(string.punctuation)
    modified_test_point2 = []
    for word in modified_test_point:
        modified_test_point2.append(
            ''.join([i for i in word if not i.isdigit() and i not in punctuations]))
    
    #Step5 : remove stopwords
    neutral_words = ['could', 'might', 'would', 'may', 'shall','www', 'http', 'email', 'sent', 'send', 'subject']
    special_characters = ['+', '-', '_', '?', '<=', '>=', '>', '<',
                          '(', ')', '{', '}',  '[', ']', '"', ';', ':', '!', '*', '@', '#', '$', '%', '&', '~', ',', '.', '\ ',  '/']

    updated_stop_words = special_characters + \
        neutral_words + list(stopwords.words('english'))

    #remove the duplicate word
    modified_test_point3 = []
    for word in modified_test_point2:
        if word not in updated_stop_words and len(word) > 2:
            modified_test_point3.append(
                word.translate(string.punctuation).lower())

    #Step 6 : lemmatization
    for i in range(len(modified_test_point3)):
        modified_test_point3[i] = WordNetLemmatizer().lemmatize(modified_test_point3[i])

    return modified_test_point3


def train_model(dataset):
    rows, _ = dataset.shape 
    number_ham_mails = 0  
    number_spam_mails = 0  
    dictionary = {}

    for i in range(rows):
        if dataset.loc[i][1] == 1:
            number_spam_mails += 1
        else:
            number_ham_mails += 1

        email = dataset.loc[i][0]
        content = list(pre_processing(email))

        #counting the number of time word occur in spam and non-spam mail
        for word in content:
            if word not in dictionary:
                if dataset.loc[i][1] == 0:
                    dictionary[word] = [1, 0]
                else:
                    dictionary[word] = [0, 1]
            else:
                if dataset.loc[i][1] == 0:
                    dictionary[word][0] += 1
                else:
                    dictionary[word][1] += 1

    # Increasing the count of each word by 1 each in both the categories (as a part of Laplace smoothing) .
    for word in dictionary:
        dictionary[word][0] += 1
        dictionary[word][1] += 1

    filtered_dictionary = {}
    for word in dictionary:
            filtered_dictionary[word] = [
                dictionary[word][0], dictionary[word][1]]

    # Creating Probability table. It will store elements in form "word: [No of ham mails in which word occur/total ham mails , No of spam mails in which word occur/total spam mails]"
    probability_table = {}
    for word in filtered_dictionary:
        probability_table[word] = [filtered_dictionary[word][0] /
                                   (number_ham_mails + 1), filtered_dictionary[word][1] / (number_spam_mails + 1)]

    return probability_table              


def get_label(probability_table, content):
    """
    # Run model to determine label of processed content
    """
    probability_ham = probability_spam = 0.5
    probability_words_ham = probability_words_spam = 1e175

    for word in content:
        if word in probability_table:
            probability_words_ham *= probability_table[word][0]
            probability_words_spam *= probability_table[word][1]

    probability_ham_words = probability_ham * probability_words_ham
    probability_spam_words = probability_spam * probability_words_spam
    
    label = 1
    if probability_ham_words >= probability_spam_words:
        label = 0

    return label

