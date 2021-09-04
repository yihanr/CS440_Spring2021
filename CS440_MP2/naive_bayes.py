# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math
from math import log


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # return predicted labels of development set


    p_ham = pos_prior
    p_spam = 1 - p_ham

    ham_dict = dict()
    spam_dict = dict()

    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            for word in train_set[i]:
                if word in ham_dict.keys():
                    ham_dict[word] += 1
                else:
                    ham_dict[word] = 1
        else:
            for word in train_set[i]:
                if word in spam_dict.keys():
                    spam_dict[word] += 1
                else:
                    spam_dict[word] = 1

    k = smoothing_parameter
    N_ham = sum(ham_dict.values())
    N_spam = sum(spam_dict.values())
    X_ham = len(ham_dict.keys())
    X_spam = len(spam_dict.keys())

    dev_labels = []
    for email in dev_set:
        p_dev_ham = log(p_ham)
        p_dev_spam = log(p_spam)
        for word in email:
            if word in ham_dict.keys():
                x_count_ham = ham_dict[word]
            else:
                x_count_ham = 0
            if word in spam_dict.keys():
                x_count_spam = spam_dict[word]
            else:
                x_count_spam = 0
            p_dev_ham = p_dev_ham + log((x_count_ham + k) / (N_ham + k * X_ham))
            p_dev_spam = p_dev_spam + log((x_count_spam + k) / (N_spam + k * X_spam))
        if p_dev_ham >= p_dev_spam:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels

    return []
    