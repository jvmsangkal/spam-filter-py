import operator


class SpamHamClassifier(object):
    def __init__(self, training_set, vocabulary_size):
        self._training_set = training_set  # list of Documents
        self._build_vocabulary(vocabulary_size)

        self._ham_list = self._build_documents('ham')
        self._spam_list = self._build_documents('spam')

        self._compute_prior_probabilities()
        self._ham_likelihood = self._compute_likelihood(self.ham_list)
        self._spam_likelihood = self._compute_likelihood(self.spam_list)

    @property
    def training_set(self):
        return self._training_set

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def spam_list(self):
        return self._spam_list

    @property
    def ham_list(self):
        return self._ham_list

    @property
    def probability_spam(self):
        return self._probability_spam

    @property
    def probability_ham(self):
        return self._probability_ham

    @property
    def spam_likelihood(self):
        return self._spam_likelihood

    @property
    def ham_likelihood(self):
        return self._ham_likelihood

    def _build_vocabulary(self, vocabulary_size):
        dictionary = {}

        for data in self.training_set:
            for token in data.tokens:
                if token in dictionary:
                    dictionary[token] += 1
                else:
                    dictionary[token] = 1

        sorted_dictionary = sorted(
            dictionary.items(),
            key=operator.itemgetter(1),
            reverse=True
        )[:vocabulary_size]

        self._vocabulary = [w[0] for w in sorted_dictionary]

    def _build_documents(self, label):
        return [self._vectorize(d)
                for d in self.training_set if d.label == label]

    def _vectorize(self, document):
        return [1 if v in document.tokens else 0
                for v in self.vocabulary]

    def _compute_prior_probabilities(self):
        self._probability_spam = len(self.spam_list)/len(self.training_set)
        self._probability_ham = len(self.ham_list)/len(self.training_set)

    def _compute_likelihood(self, documents):
        return [sum(z)/len(documents) for z in zip(documents)]

    def classify(self, document, lambda_constant):
        vector = self._vectorize(document)

        probability_document_spam = self._get_document_probability(
            vector, self.spam_likelihood)

        probability_document_ham = self._get_document_probability(
            vector, self.ham_likelihood)

        probability_spam_document = self._compute_bayes(
            probability_document_spam,
            probability_document_ham,
            lambda_constant
        )

        if probability_spam_document > 0.5:
            return 'spam'

        return 'ham'

    def _get_document_probability(self, vector, likelihood_list):
        res = 1

        for i, x in enumerate(vector):
            likelihood = likelihood_list[i]
            if x == 0:
                likelihood = 1 - likelihood

            res *= likelihood

        return res

    def _compute_bayes(self, p_doc_spam, p_doc_ham, lambda_constant=0):
        return ((p_doc_spam * self.probability_spam) + lambda_constant) / \
            ((p_doc_spam * self.probability_spam) +
             (p_doc_ham * self.probability_ham) +
             (lambda_constant * len(self.vocabulary)))
