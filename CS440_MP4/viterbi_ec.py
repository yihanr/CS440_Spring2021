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
Extra Credit: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

from collections import defaultdict
import math
from math import log

#Count occurrences of tags, tag pairs, tag/word pairs
def Count_occurences(train):
	tags = dict()
	words = dict()
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
				tags[tag_i] = 1
			else:
				tags[tag_i] += 1

			#collect for word
			if word_i not in words:
				words[word_i] = 1
			else:
				words[word_i] += 1

			# collect for tag_word except last element
			if tag_i not in tag_word_pair or word_i not in tag_word_pair[tag_i]:
				tag_word_pair[tag_i][word_i] = 1
			else:
				tag_word_pair[tag_i][word_i] += 1
		

	return tags, words, initial_tags, tags_pairs, tag_word_pair

#Compute smoothed probabilities and take the log of each probability
def Compute_log_probs(initial_tags, tags_pairs, tag_word_pair, smooth_parameter, hapax_word_tag, hapax_words_total):
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
		hapax_prob = hapax_word_tag[tag] / hapax_words_total
		for word in tag_word_pair[tag].keys():
			#tag_word_pair_prob[tag][word] = log((tag_word_pair[tag][word] + smooth_parameter) / (word_total_num + smooth_parameter * word_num_type))
			tag_word_pair_prob[tag][word] = log((tag_word_pair[tag][word] + hapax_prob * smooth_parameter) / (word_total_num + hapax_prob * smooth_parameter * word_num_type))
		tag_word_pair_unseen_prob[tag] = log(hapax_prob * smooth_parameter / (word_total_num + hapax_prob * smooth_parameter * word_num_type))

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
				"""
				elif sentence[i].endswith("ly"):
					if tag in emit_prob.keys() and "X+ly" in emit_prob[tag].keys():
						p =  tran_prob_max + emit_prob[tag]["X+ly"]
					else:
						p = tran_prob_max + emit_prob_unseen[tag]
				"""

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
					if sentence[i].endswith("ly") and tag in emit_prob.keys() and "X+ly" in emit_prob[tag].keys():
						p =  tran_prob_max + emit_prob[tag]["X+ly"]
					
					elif sentence[i].endswith("ing") and tag in emit_prob.keys() and "X+ing" in emit_prob[tag].keys():
						p =  tran_prob_max + emit_prob[tag]["X+ing"]

					elif sentence[i].endswith("ion") and tag in emit_prob.keys() and "X+ion" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+ion"]

					elif sentence[i].endswith("ed") and tag in emit_prob.keys() and "X+ed" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+ed"]

					elif (sentence[i].endswith("er") or sentence[i].endswith("or")) and tag in emit_prob.keys() and "X+er" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+er"]

					elif sentence[i].endswith("ment") and tag in emit_prob.keys() and "X+ment" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+ment"]

					elif (sentence[i].endswith("able") or sentence[i].endswith("ible")) and tag in emit_prob.keys() and "X+able" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+able"]

					elif sentence[i].endswith("ness") and tag in emit_prob.keys() and "X+ness" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+ness"]

					elif sentence[i].endswith("ful") and tag in emit_prob.keys() and "X+ful" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+ful"]

					elif (sentence[i].endswith("es") or sentence[i].endswith("s")) and tag in emit_prob.keys() and "X+es" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+es"]

					elif sentence[i].endswith("est") and tag in emit_prob.keys() and "X+est" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+est"]

					elif sentence[i].endswith("al") and tag in emit_prob.keys() and "X+al" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+al"]

					elif sentence[i].endswith("ame") and tag in emit_prob.keys() and "X+ame" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+ame"]

					elif (sentence[i].endswith("sh") or sentence[i].endswith("ch")) and tag in emit_prob.keys() and "X+sh" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+sh"]

					elif sentence[i].endswith("y") and tag in emit_prob.keys() and "X+y" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+y"]

					elif sentence[i].endswith("i") and tag in emit_prob.keys() and "X+i" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+i"]

					elif sentence[i].endswith("ant") and tag in emit_prob.keys() and "X+ant" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+ant"]


					elif sentence[i].startswith("dis") and tag in emit_prob.keys() and "dis+X" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["dis+X"]
					else:
						p = tran_prob_max + emit_prob_unseen[tag]
				temp[i][tag] = {'prob': p, 'prev': prev_tag}
				"""
				elif sentence[i].endswith("ly"):
					if tag in emit_prob.keys() and "X+ly" in emit_prob[tag].keys():
						p =  tran_prob_max + emit_prob[tag]["X+ly"]
					else:
						p = tran_prob_max + emit_prob_unseen[tag]

				elif sentence[i].endswith("ing"):
					if tag in emit_prob.keys() and "X+ing" in emit_prob[tag].keys():
						p =  tran_prob_max + emit_prob[tag]["X+ing"]
					else:
						p = tran_prob_max + emit_prob_unseen[tag]

				elif sentence[i].endswith("ion"):
					if tag in emit_prob.keys() and "X+ion" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+ion"]
					else:
						p = tran_prob_max + emit_prob_unseen[tag]

				elif sentence[i].endswith("ed"):
					if tag in emit_prob.keys() and "X+ed" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+ed"]
					else:
						p = tran_prob_max + emit_prob_unseen[tag]

				elif sentence[i].endswith("er"):
					if tag in emit_prob.keys() and "X+er" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+er"]
					else:
						p = tran_prob_max + emit_prob_unseen[tag]

				elif sentence[i].endswith("ment"):
					if tag in emit_prob.keys() and "X+ment" in emit_prob[tag].keys():
						p = tran_prob_max + emit_prob[tag]["X+ment"]
					else:
						p = tran_prob_max + emit_prob_unseen[tag]
				else:
					p = tran_prob_max + emit_prob_unseen[tag]
				temp[i][tag] = {'prob': p, 'prev': prev_tag}
				"""

				
				

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

def extra(temp_hapax_dict, tag_word_pair):
	ly_count = defaultdict(int)
	ing_count = defaultdict(int)
	ion_count = defaultdict(int)
	ed_count = defaultdict(int)
	er_count = defaultdict(int)
	ment_count = defaultdict(int)
	ness_count = defaultdict(int)
	able_count = defaultdict(int)
	ful_count = defaultdict(int)
	es_count = defaultdict(int)
	est_count = defaultdict(int)
	al_count = defaultdict(int)
	ame_count = defaultdict(int)
	sh_count = defaultdict(int)
	y_count = defaultdict(int)
	t_count = defaultdict(int)
	i_count = defaultdict(int)
	ant_count = defaultdict(int)


	dis_count = defaultdict(int)

	hapax_word_tag = defaultdict(int)


	for i in range(len(temp_hapax_dict)):
		if temp_hapax_dict[i][1].endswith("ly"):
			ly_count[temp_hapax_dict[i][0]] += 0.0003
		
		elif temp_hapax_dict[i][1].endswith("ing"):
			ing_count[temp_hapax_dict[i][0]] += 0.0003

		elif temp_hapax_dict[i][1].endswith("sion"):
			ion_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("ed"):
			ed_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("er") or temp_hapax_dict[i][1].endswith("or"):
			er_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("ment"):
			ment_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("ness"):
			ness_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("able") or temp_hapax_dict[i][1].endswith("ible"):
			able_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("ful"):
			ful_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("es") or temp_hapax_dict[i][1].endswith("s"):
			es_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("est"):
			est_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("al"):
			al_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("ame"):
			ame_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("sh") or temp_hapax_dict[i][1].endswith("ch"):
			sh_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("y"):
			y_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("i"):
			i_count[temp_hapax_dict[i][0]] += 0.0003
		elif temp_hapax_dict[i][1].endswith("ant"):
			ant_count[temp_hapax_dict[i][0]] += 0.0003

		elif temp_hapax_dict[i][1].startswith("dis"):
			dis_count[temp_hapax_dict[i][0]] += 0.0003
		else:
			hapax_word_tag[temp_hapax_dict[i][0]] += 1



		


	for tag in ly_count.keys():
		tag_word_pair[tag]["X+ly"] = ly_count[tag]
	

	for tag in ing_count.keys():
		tag_word_pair[tag]["X+ing"] = ing_count[tag]


	for tag in ion_count.keys():
		tag_word_pair[tag]["X+ion"] = ion_count[tag]

	for tag in ed_count.keys():
		tag_word_pair[tag]["X+ed"] = ed_count[tag]

	for tag in er_count.keys():
		tag_word_pair[tag]["X+er"] = er_count[tag]	

	for tag in ment_count.keys():
		tag_word_pair[tag]["X+ment"] = ment_count[tag]

	for tag in ness_count.keys():
		tag_word_pair[tag]["X+ness"] = ness_count[tag]

	for tag in able_count.keys():
		tag_word_pair[tag]["X+able"] = able_count[tag]

	for tag in ful_count.keys():
		tag_word_pair[tag]["X+ful"] = ful_count[tag]

	for tag in es_count.keys():
		tag_word_pair[tag]["X+es"] = es_count[tag]

	for tag in est_count.keys():
		tag_word_pair[tag]["X+est"] = est_count[tag]

	for tag in al_count.keys():
		tag_word_pair[tag]["X+al"] = al_count[tag]

	for tag in ame_count.keys():
		tag_word_pair[tag]["X+ame"] = ame_count[tag]

	for tag in sh_count.keys():
		tag_word_pair[tag]["X+sh"] = sh_count[tag]

	for tag in y_count.keys():
		tag_word_pair[tag]["X+y"] = y_count[tag]

	for tag in i_count.keys():
		tag_word_pair[tag]["X+i"] = i_count[tag]

	for tag in ant_count.keys():
		tag_word_pair[tag]["X+ant"] = ant_count[tag]

	for tag in dis_count.keys():
		tag_word_pair[tag]["dis+X"] = dis_count[tag]

	

	return tag_word_pair, hapax_word_tag


def viterbi_ec(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    result = []
    smooth_parameter = 0.00000001
    tags, words, initial_tags, tags_pairs, tag_word_pair = Count_occurences(train)

    #collect hapax words for tag
    hapax_word_tag = dict()
    temp_hapax_dict = list()
    for tag in tag_word_pair.keys():
    	for word in tag_word_pair[tag].keys():
    		if words[word] == 1:
    			if (tag, word) not in temp_hapax_dict:
    				temp_hapax_dict.append((tag, word))
    			if tag not in hapax_word_tag:
    				hapax_word_tag[tag] = 1
    			else:
    				hapax_word_tag[tag] += 1

    tag_word_pair, hapax_word_tag = extra(temp_hapax_dict, tag_word_pair)
    hapax_words_total = sum(hapax_word_tag.values())
    # set default value for other tag
    for tag in tags.keys():
    	if tag not in hapax_word_tag:
    		hapax_word_tag[tag] = 0.0000001

    #tag_word_pair = extra(temp_hapax_dict, tag_word_pair)

    init_prob, init_prob_unseen, tran_prob, tran_prob_unseen, emit_prob, emit_prob_unseen = Compute_log_probs(initial_tags, tags_pairs, tag_word_pair, smooth_parameter, hapax_word_tag, hapax_words_total)

    for sentence in test:
        trellis = Construct_trellis(sentence, init_prob, tran_prob, emit_prob, tran_prob_unseen, emit_prob_unseen)
        sentence_list = Find_best_path(sentence, trellis)
        result.append(sentence_list)

    return result
    return []