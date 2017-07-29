import numpy
import tensorflow as tf
from itertools import combinations

##### PARAMS
rseed = 0
num_train = 1000

##### END PARAMS

numpy.random.seed(rseed)

people = ["woman", "man", "boy", "girl"] 
animals = [ "dog", "cat"]
agents = people + animals
objects = ["chair", "sofa"] 

adjectives = ["young", "old", "short", "tall"]

relations = {"in front of": "behind", "on": "underneath"}
#symmetrize
for key in relations.keys():
    relations[relations[key]] = key

positions = 2
levels = 2
identifiers = len(agents) + len(objects)

vocabulary = relations.keys() + agents + objects + adjectives + ["PAD"]
num_words = len(vocabulary)

sentences = []
#in front / behind for agents
for agent_1, agent_2 in combinations(agents, 2):
    for adjective_1 in adjectives:
	for adjective_2 in adjectives:
		# two synonym pairs
		sentences.append([adjective_1, agent_1, "in front of", adjective_2, agent_2]) 
		sentences.append([adjective_1, agent_2, "behind", adjective_2, agent_2]) 
		sentences.append([adjective_1, agent_2, "in front of", adjective_2, agent_2]) 
		sentences.append([adjective_1, agent_1, "behind", adjective_2, agent_2]) 

for agent in agents:
    for adjective_1 in adjectives:
	for target in objects:
		sentences.append([adjective_1, agent, "in front of", target])
		sentences.append([adjective_1, agent, "behind", target]) 
		sentences.append([target,"in front of", adjective_1, agent]) 
		sentences.append([target,"behind", adjective_1, agent]) 
		sentences.append([adjective_1, agent, "on", target]) 
		sentences.append([target,"underneath", adjective_1, agent]) 


def pad(l, target_length, pad_element):
    """Pads list l to target length if not already there"""
    return l+(target_length-len(l))*[pad_element]

max_length = max(map(len, sentences))
sentences = [pad(sentence, max_length, "PAD") for sentence in sentences] 

numpy.random.shuffle(sentences)
train_sentences = sentences[:1000]
test_sentences = sentences[1000:]


