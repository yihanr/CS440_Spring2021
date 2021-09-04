# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

from collections import defaultdict
import math
from math import log

#Count occurrences of tags, tag pairs, tag/word pairs
def Count_occurences(train):
	tags = []
	initial_tags = dict()
	tags_pairs = defaultdict(dict)
	tag_word_pair = defaultdict(dict)

	for sentence in train:
		# count number of occurence for each tag to be the first tag of a sentence
		first_tag = sentence[0][1]
		if first_tag not in initial_tags:
			initial_tags[first_tag] = 1
		else:
			initial_tags[first_tag] += 1
		# count occurence of tags_pairs and tag_word_pair
		n = len(sentence)
		# since we will use i+1, collect last tag_word after for loop(1,n-2)
		for i in range(n):
			word_i = sentence[i][0]
			tag_i = sentence[i][1]
			
			if(i < n-1):
				word_j = sentence[i+1][0]
				tag_j = sentence[i+1][1]
				# collect for tags_pair
				if tag_i not in tags_pairs or tag_j not in tags_pairs[tag_i]:
					tags_pairs[tag_i][tag_j] = 1
				else:
					tags_pairs[tag_i][tag_j] += 1
			# collect for tags
			if tag_i not in tags:
				tags.append(tag_i)

			# collect for tag_word except last element
			if tag_i not in tag_word_pair or word_i not in tag_word_pair[tag_i]:
				tag_word_pair[tag_i][word_i] = 1
			else:
				tag_word_pair[tag_i][word_i] += 1
		
	num_tag = len(tags)
	return num_tag, initial_tags, tags_pairs, tag_word_pair

#Compute smoothed probabilities and take the log of each probability
def Compute_log_probs(initial_tags, tags_pairs, tag_word_pair, smooth_parameter):
	# compute initial probabilities
	inital_prob = dict()
	initial_tags_num_type = len(initial_tags.keys())
	initial_tags_total_num = sum(initial_tags.values())
	for tag in initial_tags.keys():
		inital_prob[tag] = log((initial_tags[tag] + smooth_parameter) / (initial_tags_total_num + smooth_parameter * initial_tags_num_type))
	initial_unseen_prob = log(smooth_parameter / (initial_tags_total_num + smooth_parameter * initial_tags_num_type))
	
	# compute transition probabilities 
	tags_pairs_prob = defaultdict(dict)
	tags_pairs_unseen_prob = dict()
	for tag_prev in tags_pairs.keys():
		tag_next_num_type = len(tags_pairs[tag_prev].keys())
		tag_next_total_num = sum(tags_pairs[tag_prev].values())
		for tag_next in tags_pairs[tag_prev].keys():
			tags_pairs_prob[tag_prev][tag_next] = log((tags_pairs[tag_prev][tag_next] + smooth_parameter) / (tag_next_total_num + smooth_parameter * tag_next_num_type))
		tags_pairs_unseen_prob[tag_prev] = log(smooth_parameter / (tag_next_total_num + smooth_parameter * tag_next_num_type))
	tags_pairs_unseen_prob['END'] = log(smooth_parameter / (tag_next_total_num + smooth_parameter * tag_next_num_type))
	
	# compute emission probabilities
	tag_word_pair_prob = defaultdict(dict)
	tag_word_pair_unseen_prob = dict()
	for tag in tag_word_pair.keys():
		word_num_type = len(tag_word_pair[tag].keys())
		word_total_num = sum(tag_word_pair[tag].values())
		for word in tag_word_pair[tag].keys():
			tag_word_pair_prob[tag][word] = log((tag_word_pair[tag][word] + smooth_parameter) / (word_total_num + smooth_parameter * word_num_type))
		tag_word_pair_unseen_prob[tag] = log(smooth_parameter / (word_total_num + smooth_parameter * word_num_type))

	return inital_prob, initial_unseen_prob, tags_pairs_prob, tags_pairs_unseen_prob, tag_word_pair_prob, tag_word_pair_unseen_prob

#Construct the trellis.
def Construct_trellis(sentence, init_prob, tran_prob, emit_prob, tran_prob_unseen, emit_prob_unseen):

	temp = [{}]
	for i in range(len(sentence)):
		if i == 0:
			for tag in init_prob.keys():
				if sentence[i] in emit_prob[tag]:
					p = init_prob[tag] + emit_prob[tag][sentence[i]]
				else:
					p = init_prob[tag] + emit_prob_unseen[tag]
				temp[i][tag] = {'prob': p, 'prev': None}
		else:
			temp.append({})
			for tag in emit_prob.keys():
				tran_prob_max = -math.inf
				for temp_tag in temp[i-1].keys():
					if tag in tran_prob[temp_tag]:
						temp_p = temp[i-1][temp_tag]['prob'] + tran_prob[temp_tag][tag]
					else:
						temp_p = temp[i-1][temp_tag]['prob'] + tran_prob_unseen[temp_tag]
					if temp_p > tran_prob_max:
						tran_prob_max = temp_p
						prev_tag = temp_tag
				if sentence[i] in emit_prob[tag]:
					p = tran_prob_max + emit_prob[tag][sentence[i]]
				else:
					p = tran_prob_max + emit_prob_unseen[tag]
				temp[i][tag] = {'prob': p, 'prev': prev_tag}

	return temp

# find the best path through the trellis
def Find_best_path(sentence, trellis):
    best_path_sentence = []
    n = len(sentence)
    max_prob = -math.inf
    chosen_tag = None
    for tag in trellis[n - 1]:
        if trellis[n-1][tag]['prob'] > max_prob:
        	max_prob = trellis[n-1][tag]['prob']
        	chosen_tag = tag
    tag_path = [chosen_tag]
    k = n - 2
    for i in range(k, -1, -1):
        chosen_tag = trellis[i+1][chosen_tag]['prev']
        tag_path.append(chosen_tag)
        	
    tag_path = tag_path[::-1]
    for i in range(n):
        best_path_sentence.append((sentence[i], tag_path[i]))
    return best_path_sentence


def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    result = []
    smooth_parameter = 0.0001
    num_tag, initial_tags, tags_pairs, tag_word_pair = Count_occurences(train)
    init_prob, init_prob_unseen, tran_prob, tran_prob_unseen, emit_prob, emit_prob_unseen = Compute_log_probs(initial_tags, tags_pairs, tag_word_pair, smooth_parameter)

    for sentence in test:
        trellis = Construct_trellis(sentence, init_prob, tran_prob, emit_prob, tran_prob_unseen, emit_prob_unseen)
        sentence_list = Find_best_path(sentence, trellis)
        result.append(sentence_list)

    return result
    return []

    