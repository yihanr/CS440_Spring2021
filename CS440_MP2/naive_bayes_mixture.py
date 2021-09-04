# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math
from math import log



def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    # return predicted labels of development set


    p_ham = pos_prior
    p_spam = 1 - p_ham

    ham_uni_dict = dict()
    spam_uni_dict = dict()

    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            for word in train_set[i]:
                if word in ham_uni_dict.keys():
                    ham_uni_dict[word] += 1
                else:
                    ham_uni_dict[word] = 1
        else:
            for word in train_set[i]:
                if word in spam_uni_dict.keys():
                    spam_uni_dict[word] += 1
                else:
                    spam_uni_dict[word] = 1

    k_uni = unigram_smoothing_parameter
    N_uni_ham = sum(ham_uni_dict.values())
    N_uni_spam = sum(spam_uni_dict.values())
    X_uni_ham = len(ham_uni_dict.keys())
    X_uni_spam = len(spam_uni_dict.keys())

    ham_bi_dict = dict()
    spam_bi_dict = dict()
    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            for j in range(len(train_set[i]) - 1):
                bi_words = train_set[i][j] + train_set[i][j+1]
                if bi_words in ham_bi_dict.keys():
                    ham_bi_dict[bi_words] += 1
                else:
                    ham_bi_dict[bi_words] = 1
        else:
            for j in range(len(train_set[i]) - 1):
                bi_words = train_set[i][j] + train_set[i][j+1]
                if bi_words in spam_bi_dict.keys():
                    spam_bi_dict[bi_words] += 1
                else:
                    spam_bi_dict[bi_words] = 1

    k_bi = bigram_smoothing_parameter
    N_bi_ham = sum(ham_bi_dict.values())
    N_bi_spam = sum(spam_bi_dict.values())
    X_bi_ham = len(ham_bi_dict.keys())
    X_bi_spam = len(spam_bi_dict.keys())

    dev_labels = []
    for email in dev_set:
        p_dev_ham_uni = log(p_ham)
        p_dev_spam_uni = log(p_spam)
        for word in email:
            if word in ham_uni_dict.keys():
                x_counts_ham = ham_uni_dict[word]
            else:
                x_counts_ham = 0
            if word in spam_uni_dict.keys():
                x_counts_spam = spam_uni_dict[word]
            else:
                x_counts_spam = 0
            p_dev_ham_uni = p_dev_ham_uni + log((x_counts_ham + k_uni) / (N_uni_ham + k_uni * X_uni_ham))
            p_dev_spam_uni = p_dev_spam_uni + log((x_counts_spam + k_uni) / (N_uni_spam + k_uni * X_uni_spam))
        p_dev_ham_bi = log(p_ham)
        p_dev_spam_bi = log(p_spam)
        for i in range(len(email) - 1):
            bi_words = email[i] + email[i+1]
            if bi_words in ham_bi_dict.keys():
                x_counts_ham = ham_bi_dict[bi_words]
            else:
                x_counts_ham = 0
            if bi_words in spam_bi_dict.keys():
                x_counts_spam = spam_bi_dict[bi_words]
            else:
                x_counts_spam = 0
            p_dev_ham_bi = p_dev_ham_bi + log((x_counts_ham + k_bi) / (N_bi_ham + k_bi * X_bi_ham))
            p_dev_spam_bi = p_dev_spam_bi + log((x_counts_spam + k_bi) / (N_bi_spam + k_bi * X_bi_spam))
        p_dev_ham = (1 - bigram_lambda) * p_dev_ham_uni + bigram_lambda * p_dev_ham_bi
        p_dev_spam = (1 - bigram_lambda) * p_dev_spam_uni + bigram_lambda * p_dev_spam_bi
        if p_dev_ham >= p_dev_spam:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels




    return []

