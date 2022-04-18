import os
import nltk
import train as tr
import pandas as pd

def load_nltk_package():
    """
    Download the package of the wordnet and stopwords corpos
    """
    nltk.download('stopwords')
    nltk.download('wordnet')


def get_test_mails(dir_path):
    """
    Get the all test data ( email to classify)

    ------------------
    input : directory path

    ------------------
    return all the list of emails
    """

    test_mails = []

    for i in range(len(os.listdir(dir_path))):
        path = dir_path + '/' + os.listdir(dir_path)[i]
        if not os.path.isfile(path):
            continue
        file_pointer = open(path, encoding='utf-8')
        mail = file_pointer.read()
        file_name = os.listdir(dir_path)[i][:-4]
        test_mails.append([file_name, mail])

    return test_mails


def get_training_dataset(file_path):
    """
    Read the training data from the file

    -----------------------
    input : training file path
    -----------------------
    output : dataset
    """
    dataset = pd.read_csv(
        "../dataset/training_dataset.csv", encoding="latin-1")
    dataset.dropna(axis=0, how='any', thresh=None, subset=None,
                   inplace=True)  # remove the row contains NA valus

    return dataset


def print_output_file(spam_results, output_file_path):
    """
    Print the output to the file
    """

    file_pointer = open(output_file_path, 'w')
    file_pointer.write("Email, Label \n")

    for line in spam_results:
        file_pointer.write(line+'\n')
    
    print("Output generated successfully . . .!")
    file_pointer.close()


def main():

    training_file_path = "../dataset/training_dataset.csv"
    output_file_path = "../output/output.csv"
    test_email_folder = "../test"

    # 1.Data Pre-processing

    # Load the nltk packages
    load_nltk_package()
    # Get the dataset for training
    dataset = get_training_dataset(training_file_path)

    # 2.Model training
    print("Model training is started . . . . .")
    model = tr.train_model(dataset)
    print("Model training is completed.")

    # 3.Model Evaluation : This is for the given requirment ( read the email files)
    print("Model Evaluation is started. . ...")
    test_mails = get_test_mails(test_email_folder)
    output_lines = []
    for mail in test_mails:
        name = mail[0]
        content = mail[1]

        # preprocessing the content
        pre_processed_content = tr.pre_processing(content)
        label = tr.get_label(model, pre_processed_content)
        output_lines.append(name + ',' + str(label))

    print_output_file(output_lines, output_file_path)

    print("Model evaluation is completed.") 

if __name__ == "__main__":
    main()
