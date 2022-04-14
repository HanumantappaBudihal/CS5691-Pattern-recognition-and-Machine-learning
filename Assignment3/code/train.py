import utilities as utl

def train_model(dataset):
    """
    """
    rows, _ = dataset.shape  #Rows and cols
    number_ham_mails = 0     #Numbers of Ham mails
    number_spam_mails = 0    #Numbers of Spam mails
    dictionary = {}

    for i in range(rows):
        if dataset.loc[i][1] == 1:
            number_spam_mails += 1
        else:
            number_ham_mails += 1

        email = dataset.loc[i][0]
        content = list(set(utl.lemmatize(utl.remove_stopwords(utl.remove_numbers_punctuations(
            utl.remove_formatting(utl.remove_mail_id(utl.get_tokens(email))))))))

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
        if (dictionary[word][0] / dictionary[word][1]) > 1.7 or (dictionary[word][1] / dictionary[word][0]) > 1.7:
            filtered_dictionary[word] = [
                dictionary[word][0], dictionary[word][1]]

    # Creating Probability table. It will store elements in form "word: [No of ham mails in which word occur/total ham mails , No of spam mails in which word occur/total spam mails]"
    probability_table = {}
    for word in filtered_dictionary:
        probability_table[word] = [filtered_dictionary[word][0] /
                                   (number_ham_mails + 1), filtered_dictionary[word][1] / (number_spam_mails + 1)]

    return probability_table


def pre_process(content):
    """
    # Preprocess raw mail content
    """
    return list(set(utl.lemmatize(utl.remove_stopwords(utl.remove_numbers_punctuations(utl.remove_formatting(utl.remove_mail_id(utl.get_tokens(content))))))))


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
