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
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    temp = dict()
    tags = dict()
    #count = 0
    result = []
    for sentence in train:
    	for word in sentence:
    		if word[0] not in temp.keys():
    			temp[word[0]] = [[word[1]], [1]]
    		else:
    			if word[1] in temp[word[0]][0]:
    				temp[word[0]][1][temp[word[0]][0].index(word[1])] += 1
    			else:
    				temp[word[0]][0].append(word[1])
    				temp[word[0]][1].append(1)
    		if word[1] in tags.keys():
    			tags[word[1]] += 1
    		else:
    			tags[word[1]] = 1
    		#count += 1
    all_tags = list(tags.keys())
    tags_count = list(tags.values())
    largest_count = max(tags_count)
    index = tags_count.index(largest_count)
    most_often_tag = all_tags[index]
    for sentence in test:
    	sentence_list = []
    	for word in sentence:
    		if word in temp.keys():
    			tag = temp[word][0][temp[word][1].index(max(temp[word][1]))]
    			sentence_list.append((word, tag))
    		else:
    			sentence_list.append((word, most_often_tag))
    	result.append(sentence_list)

    #print(train[1][1])
    return result
    return []