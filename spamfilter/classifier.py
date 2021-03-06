from functools import reduce
from collections import Counter
import math
import operator

import numpy as np


class SpamHamClassifier(object):
    def __init__(self, training_data, vocabulary_size,
                 compute_mutual_information, lambda_constant=0):
        self._num_training_data = len(training_data)
        self._lambda_constant = lambda_constant

        self._num_ham_documents = 0
        self._num_spam_documents = 0
        self._ham_counter = Counter()
        self._spam_counter = Counter()

        vocabulary = Counter()
        for data in training_data:
            counter = Counter(data.tokens)
            vocabulary.update(counter)

            vectorized = self._vectorize(counter)
            if data.label == 'ham':
                self._num_ham_documents += 1
                self._ham_counter.update(vectorized)
            elif data.label == 'spam':
                self._num_spam_documents += 1
                self._spam_counter.update(vectorized)

        self._probability_ham = np.divide(
            self.num_ham_documents,
            self.num_training_data
        )

        self._probability_spam = np.divide(
            self.num_spam_documents,
            self.num_training_data
        )

        if compute_mutual_information:
            word_mi = {}

            for word, frequency in vocabulary.items():
                pwordspam = self.spam_counter[word] / len(training_data)
                pwordham = self.ham_counter[word] / len(training_data)
                pnotwordspam = (len(training_data) - self.spam_counter[word]) / len(training_data)
                pnotwordham = (len(training_data) - self.ham_counter[word]) / len(training_data)
                pword = frequency / len(training_data)
                pnotword = (len(training_data) - frequency) / len(training_data)
                mi = np.sum([
                    np.multiply(
                        pwordham,
                        np.log(
                            np.divide(
                                pwordham,
                                np.multiply(pword, self.probability_ham)
                            )
                        )
                    ),
                    np.multiply(
                        pwordspam,
                        np.log(
                            np.divide(
                                pwordspam,
                                np.multiply(pword, self.probability_spam)
                            )
                        )
                    ),
                    np.multiply(
                        pnotwordham,
                        np.log(
                            np.divide(
                                pnotwordspam,
                                np.multiply(pnotword, self.probability_ham)
                            )
                        )
                    ),
                    np.multiply(
                        pnotwordspam,
                        np.log(
                            np.divide(
                                pnotwordspam,
                                np.multiply(pnotword, self.probability_spam)
                            )
                        )
                    )
                ])

                word_mi[word] = mi

            word_mi = sorted(
                        word_mi.items(), key=lambda kv: kv[1], reverse=True)
            vocabulary = word_mi[:vocabulary_size]
        else:
            vocabulary = vocabulary.most_common(vocabulary_size)

        self._vocabulary = [v[0]
                            for v in vocabulary]

        self._ham_counter = Counter({
            k: v for k, v in self.ham_counter.items() if k in self.vocabulary
        })
        self._spam_counter = Counter({
            k: v for k, v in self.spam_counter.items() if k in self.vocabulary
        })

    @property
    def num_training_data(self):
        return self._num_training_data

    @property
    def num_spam_documents(self):
        return self._num_spam_documents

    @property
    def num_ham_documents(self):
        return self._num_ham_documents

    @property
    def lambda_constant(self):
        return self._lambda_constant

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def spam_counter(self):
        return self._spam_counter

    @property
    def ham_counter(self):
        return self._ham_counter

    @property
    def probability_spam(self):
        return self._probability_spam

    @property
    def probability_ham(self):
        return self._probability_ham

    def _vectorize(self, counter):
        return Counter({x: 1 for x in counter})

    def classify(self, document):
        vector = self._vectorize(document.tokens)

        document_likelihood_spam = self._compute_likelihood(
            vector,
            self.num_spam_documents,
            self.spam_counter
        )

        document_likelihood_ham = self._compute_likelihood(
            vector,
            self.num_ham_documents,
            self.ham_counter
        )

        probability_ham_document = self._compute_bayes(
            document_likelihood_ham,
            document_likelihood_spam
        )

        if probability_ham_document >= 0.5:
            return 'ham'

        return 'spam'

    def _compute_likelihood(self, document, label_total, labelled_counter):
        tmp = []

        vocabulary = self.vocabulary
        if self.lambda_constant:
            vocabulary = list(document.keys())

        for word in vocabulary:
            count = labelled_counter[word]
            if not document[word]:
                count = label_total - labelled_counter[word]

            likelihood = np.divide(
                np.add(count, self.lambda_constant),
                np.add(
                    label_total,
                    np.multiply(self.lambda_constant, len(self.vocabulary))
                )
            )

            if likelihood == 0:
                return 0.0

            tmp.append(np.log(likelihood))

        return np.exp(np.sum(tmp), dtype=np.float128)

    def _compute_bayes(self, ham_likelihood, spam_likelihood):
        return np.divide(
            np.multiply(ham_likelihood, self.probability_ham),
            np.add(
                np.multiply(ham_likelihood, self.probability_ham),
                np.multiply(spam_likelihood, self.probability_spam)
            )
        )
