import numpy
import tensorflow as tf
from itertools import combinations, product

##### PARAMS
rseed = 0
num_train = 1000

nhidden_representation = 30

##### END PARAMS

numpy.random.seed(rseed)

people = ["woman", "man"] #, "boy", "girl"] 
animals = [ "dog", "cat"]
agents = people # + animals
objects = ["chair"] #, "sofa"] 
adjectives = ["young", "old"]#, "short", "tall"]
identifiers = numpy.array(list(product(adjectives, agents)) + objects) # order will be used for one-hot IDs  
num_identifiers = len(identifiers)
empty_identifier = numpy.zeros(num_identifiers)


relations = {"in front of": "behind", "on": "underneath"}
#symmetrize
for key in relations.keys():
    relations[relations[key]] = key

positions = 2
levels = 2

vocabulary = relations.keys() + agents + objects + adjectives + ["GO", "PAD"]
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

sentence_length = max(map(len, sentences))
sentences = [pad(sentence, sentence_length, "PAD") for sentence in sentences] 

def sentence_to_world(sentence):
    """Takes a sentence, returns a corresponding world"""
    def _build_world(identifier_1, identifier_2, relation):
	identifier_1 = [ident == identifier_1 for ident in identifiers] 
	identifier_2 = [ident == identifier_2 for ident in identifiers] 
	if relation == "in front of":
	    world = numpy.array([[identifier_1, empty_identifier], [identifier_2, empty_identifier]])
	elif relation == "behind":
	    world = numpy.array([[identifier_2, empty_identifier], [identifier_1, empty_identifier]])
	elif relation == "on":
	    #TODO: something more elegant
	    if numpy.random.rand() < 0.5:
		world = numpy.array([[identifier_2, identifier_1], [empty_identifier, empty_identifier]])
	    else:
		world = numpy.array([[empty_identifier, empty_identifier], [identifier_2, identifier_1]])
	elif relation == "underneath":
	    #TODO: something more elegant
	    if numpy.random.rand() < 0.5:
		world = numpy.array([[identifier_1, identifier_2], [empty_identifier, empty_identifier]])
	    else:
		world = numpy.array([[empty_identifier, empty_identifier], [identifier_1, identifier_2]])
	return world.flatten()

    if sentence[-2] in objects:
	identifier_1 = (sentence[0], sentence[1])		 
	identifier_2 = sentence[-2]
	relation = sentence[2]
    elif sentence[0] in objects:
	identifier_1 = sentence[0]
	identifier_2 = (sentence[-3], sentence[-2])		 
	relation = sentence[1]
    else: # two agents 
	identifier_1 = (sentence[0], sentence[1])		 
	identifier_2 = (sentence[-2], sentence[-1])		 
	relation = sentence[2]

    return _build_world(identifier_1, identifier_2, relation)

def world_to_sentence(world):
    """Takes a world, returns a corresponding sentence"""
    world = world.reshape((positions, levels, num_identifiers))
    occupied = numpy.zeros((positions, levels), dtype=numpy.bool)
    these_identifiers = []
    for position in xrange(positions):
	for level in xrange(levels):
	    if numpy.sum(world[position, level] == 1) == 1:
		occupied[position, level] = True
		this_identifier = identifiers[world[position, level] == 1.][0]
		if type(this_identifier) != tuple:
		    this_identifier = (this_identifier,)
		these_identifiers.append(this_identifier)

    if numpy.any(numpy.sum(occupied, 1) == 2): # on/under
	if numpy.random.rand() < 0.5:
	    sentence = list(these_identifiers[0]) + ["under"] + list(these_identifiers[1])
	else:
	    sentence = list(these_identifiers[1]) + ["on"] + list(these_identifiers[0])
    else: # front/behind
	if numpy.random.rand() < 0.5:
	    sentence = list(these_identifiers[0]) + ["in front of"] + list(these_identifiers[1])
	else:
	    sentence = list(these_identifiers[1]) + ["behind"] + list(these_identifiers[0])
    return pad(sentence, max_length, "PAD")

numpy.random.shuffle(sentences)
worlds = map(build_world, sentences)
#train_sentences = sentences[:num_train]
#test_sentences = sentences[num_train:]
#train_worlds = worlds[:num_train]
#test_worlds = worlds[num_train:]

class model:
    def __init__(self, vocab):
	self.vocab = vocab
	vocab_size = len(vocab)
	self.description_targets_ph = tf.placeholder(tf.int32, shape=[sentence_length, self.vocab_size])
        self.description_inputs_ph = tf.placeholder(tf.int32, shape=[sentence_length, 1])
	self.representation_ph = tf.placeholder(tf.float32, shape=[nhidden_representation,])
	self.visual_input_ph = tf.placeholder(tf.float32, shape=[num_levels, num_identifiers, num_positions])
	self.visual_target_ph = tf.placeholder(tf.float32, shape=[num_levels, num_identifiers, num_positions])


	with tf.variable_scope('description_input'):
	    description_inputs_reversed = tf.reverse(self.description_inputs_ph,[True,False]) 	
	    self.word_embeddings = embeddings = tf.Variable(tf.random_uniform([self.vocab_size, embedding_size], -1, 1))
	    self.embedded_description_inputs = tf.nn.embedding_lookup(self.word_embeddings, description_inputs_reversed)
	    __, self.encoded_descr_input = tf.nn.dynamic_rnn(tf.nn.rnn_cell.BasicRNNCell(nhidden_representation), self.embedded_description_inputs, dtype=tf.float32, time_major=True)		
	
	with tf.variable_scope('description_output') as scope:
	    self.Wd = tf.Variable(tf.random_normal([embedding_size, nhidden_representation], 0, 1./nhidden_representation))
	    self.bd = tf.Variable(tf.zeros([embedding_size,]))

            output_cell = tf.nn.rnn_cell.BasicRNNCell(nhidden_representation)
            curr_decoder_state = tf.transpose(self.representation_h)
            curr_decoder_input = tf.reshape(tf.nn.embedding_lookup(self.word_embeddings, vocab_dict["GO"]),[1, embedding_size])
            outputs = []
            for i in xrange(sentence_length):
                if i > 0:
                   scope.reuse_variables()
                output , curr_decoder_state = output_cell(curr_decoder_input, curr_decoder_state)
                output = tf.nn.tanh(tf.matmul(self.W4d,tf.transpose(output))+self.b4d)
                outputs.append(output)
                curr_decoder_input = tf.transpose(output)

	    self.output_logits = [tf.matmul(self.word_embeddings,output) for output in outputs]
	    self.descr_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.transpose(tf.concat(1, self.output_logits)), self.description_targets_ph))

	with tf.variable_scope('visual_input'): 
	    W1_input = tf.Variable(tf.random_normal([num_identifiers*num_positions*num_levels, nhidden_representation], 0., 1./nhidden_representation))
	    b1_input = tf.Variable(tf.zeros([nhidden_representation,]))
	    W2_input = tf.Variable(tf.random_normal([nhidden_representation, nhidden_representation], 0., 1./nhidden_representation))
	    b2_input = tf.Variable(tf.zeros([nhidden_representation,]))
	    W3_input = tf.Variable(tf.random_normal([nhidden_representation, nhidden_representation], 0., 1./nhidden_representation))
	    b3_input = tf.Variable(tf.zeros([nhidden_representation,]))

	    visual_input = tf.reshape(self.visual_input_ph, [-1,])
	    net = tf.nn.tanh(tf.nn.matmul(W1_input, visual_input) + b1_input)
	    net = tf.nn.tanh(tf.nn.matmul(W2_input, net) + b2_input)
	    self.encoded_visual_input = tf.nn.tanh(tf.nn.matmul(W3_input, net) + b3_input)

	with tf.variable_scope('visual_output'):
	    W1_output = tf.Variable(tf.random_normal([nhidden_representation, nhidden_representation], 0., 1./nhidden_representation))
	    b1_output = tf.Variable(tf.zeros([nhidden_representation,]))
	    W2_output = tf.Variable(tf.random_normal([nhidden_representation, nhidden_representation], 0., 1./nhidden_representation))
	    b2_output = tf.Variable(tf.zeros([nhidden_representation,]))
	    W3_output = tf.Variable(tf.random_normal([num_identifiers*num_positions*num_levels, nhidden_representation], 0., 1./nhidden_representation))
	    b3_output = tf.Variable(tf.zeros([num_identifiers*num_positions*num_levels,]))

	    net = tf.nn.tanh(tf.nn.matmul(W1_output, visual_output) + b1_output)
	    net = tf.nn.tanh(tf.nn.matmul(W2_output, net) + b2_output)
	    self.visual_output = tf.reshape(tf.nn.tanh(tf.nn.matmul(W3_output, net) + b3_output), [num_positions, num_levels, num_identifiers])

	    self.visual_loss = tf.nn.l2_loss(self.visual_output - self.visual_target_ph) 

