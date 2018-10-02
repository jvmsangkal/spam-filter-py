# -*- coding: latin-1 -*-
from spamfilter import Document
from spamfilter import SpamHamClassifier

import os
import nltk
import time
import argparse


def main(args):
    nltk.download('stopwords')

    training_set = []
    test_set = []

    labels_dir = os.path.dirname(os.path.abspath(args.labels))
    data_dir = os.path.join(labels_dir, 'data')

    with open(args.labels) as labels:
        for l in labels:
            label, document_path = l.strip().split(' ')
            folder_index = int(document_path.split('/')[2])
            document_path = os.path.join(data_dir, document_path)

            with open(
                 document_path, encoding='latin-1') as raw_file:
                document = Document(
                    label,
                    raw_file,
                    args.stop_words,
                    args.stem,
                    args.gram
                )

                if folder_index < args.test_index:
                    training_set.append(document)
                else:
                    test_set.append(document)

    print('Training data: {}'.format(len(training_set)))
    print('Test data: {}'.format(len(test_set)))
    print('Constructing vocabulary from top {} words..'.format(
        args.vocabulary_size))

    classifier = SpamHamClassifier(
        training_set,
        args.vocabulary_size,
        args.compute_mutual_information,
        args.lambda_constant
    )

    print('Vocabulary: {}'.format(', '.join(classifier.vocabulary)))

    print('Number of Spam documents from training data: {}'.format(
        classifier.num_spam_documents))
    print('Number of Ham documents from training data: {}'.format(
        classifier.num_ham_documents))
    print('P(w=S) = {}'.format(classifier.probability_spam))
    print('P(w=H) = {}'.format(classifier.probability_ham))

    tp = 0
    fp = 0
    fn = 0
    print('Running tests..')
    for test in test_set:
        result = classifier.classify(test)

        if test.label == 'spam' and result == 'spam':
            tp += 1
        elif test.label == 'ham' and result == 'spam':
            fp += 1
        elif test.label == 'spam' and result == 'ham':
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spam Filter')

    parser.add_argument(
        '--labels',
        dest='labels',
        metavar='<path-to-labels>',
        required=True,
        help='Path to the labels file'
    )

    parser.add_argument(
        '--test-index',
        dest='test_index',
        type=int,
        default=71,
        help='Starting index of testing data'
    )

    parser.add_argument(
        '--vocabulary-size',
        dest='vocabulary_size',
        type=int,
        default=10000,
        help='Size of the vocabulary'
    )

    parser.add_argument(
        '--lambda',
        dest='lambda_constant',
        type=float,
        default=0,
        help='lambda constant to be used, λ = 2.0, 1.0, 0.5, 0.1, 0.005'
    )

    parser.add_argument(
        '--stop-words',
        dest='stop_words',
        action='store_true',
        help='flag to allow or disallow stop words when parsing'
    )

    parser.add_argument(
        '--stem',
        dest='stem',
        action='store_true',
        help="flag to use NLTK's PorterStemmer"
    )

    parser.add_argument(
        '--gram',
        dest='gram',
        type=int,
        default=1,
        help='n-gram'
    )

    parser.add_argument(
        '--compute-mutual-information',
        dest='compute_mutual_information',
        action='store_true',
        help='Flag if mutual information is computed on getting V'
    )

    args = parser.parse_args()

    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
