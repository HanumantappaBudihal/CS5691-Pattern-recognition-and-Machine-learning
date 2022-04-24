import nltk
import numpy as np
import pandas as pd
import train as tr
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix



def load_nltk_package():
    """
    Download the package of the wordnet and stopwords corpos
    """
    nltk.download('stopwords')
    nltk.download('wordnet')


def get_test_mails(test_file_path):
    test_mails = pd.read_csv(test_file_path)
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
    file_pointer.write("Email, True Label , Pred Label\n")

    for line in spam_results:
        file_pointer.write(line+'\n')

    print("Output generated successfully . . .!")
    file_pointer.close()

def plot_confusion_matrix(true_labels,pred_labels):
    conf_matrix = confusion_matrix(y_true=true_labels, y_pred=pred_labels)
    _, ax = plt.subplots(figsize=(7.5, 7.5))
    
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

def main():

    training_file_path = "../dataset/training_dataset.csv"
    output_file_path = "../output/testdata2_output.csv"
    test_emails = "../dataset/testdata2.csv"

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
    print("Testing started. . ...")
    test_mails = get_test_mails(test_emails)
    output_lines = []
    mail_counter = 0
    pred_labels = []
    true_labels=[]

    for mail in test_mails.values:
        mail_counter += 1
        content = mail[0]
        true_labels.append(mail[1])

        # preprocessing the content
        pre_processed_content = tr.pre_processing(content)
        pred_label = tr.get_label(model, pre_processed_content)
        pred_labels.append(pred_label)

        output_lines.append('email'+str(mail_counter) + ',' +
                            str(true_labels[mail_counter-1])+','+str(pred_label))

    print_output_file(output_lines, output_file_path)

    pred = np.array(pred_labels)
    true = np.array(true_labels)
    

    overall_correct = (pred == true)
    accuracy = overall_correct.sum() / overall_correct.size

    print('Accuracy :'+str(accuracy))
    plot_confusion_matrix(true_labels,pred_labels)    

if __name__ == "__main__":
    main()
