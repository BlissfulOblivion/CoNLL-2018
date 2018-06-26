
# coding: utf-8

# # CoNLL MAIN PROGRAM
# ## Encoder-Decoder with Attention
# ### Current archetype: BiGRU with Attention
# --------------------------
# #### Plan:
# 1. Finish commenting -- **DONE**
# 2. Edit evaluation functions -- **DONE**
# 3. Test with current design (get dev data and test on dev data) -- **DONE**
# 4. Incorporate tags in input -- **DONE**
# 5. Test with tags in input -- **DONE**
# 6. Convert GRU to LSTM -- **CANCELED**
# 7. Test with LSTM structure -- **CANCELED**
# 8. Edit for BiRNN structure (GRU) -- **DONE**
# 9. Test with fully developed structure
# 10. Discuss future possibilities
# --------------------------
# #### Questions:
# 1. What does .view() do? **Answer**: resizes tensors (see section 2 of tests)
# 2. How does super() work with pytorch? **Answer**: Basic inheritance
# 3. What is squeeze/unsqueeze? **Answer**: Squeeze squishes any one-sized dims in a tensor, unsqueeze adds a 1-sized dim at a given position in the tensor
# 4. What does .size() do? **Answer**: returns size of tensor
# 5. What is topk? **Answer**: Returns top k largest values of tensor along a specified dim
# --------------------------
# #### Need help with spots:
# 1. Line 38 in "# Training loop definition": How to resize tensors so they can be concatenated? Purpose: to concatenate context (morphological tags) tensor to decoder hidden state (this is where it is initiated) -- **FIXED**
# 2. In "# Decoder RNN with attention" and elsewhere: How to reshape vectors such that decoder hidden state with context included can be run through torch.bmm with encoder output without losing information? -- **FIXED**

# In[ ]:


'''      ANNOTATION KEY      '''
# S/E = self explanatory
# ??? = uncertain of meaning/purpose
# TUOP = texual update of progress


# In[ ]:


from __future__ import unicode_literals, print_function, division # ???
from io import open # For opening files
import unicodedata # To convert to ASCII -- necessary?
import string # ???
import re # For normalizing text -- necessary?
import random # For randomizing samples of data

import torch # Imports pytorch
import torch.nn as nn # Imports nn from pytorch
from torch import optim # For optimization (SGD)
import torch.nn.functional as F # For linear functions in neural nets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Sets what processor to use (GPU/CPU)


# In[ ]:


# Sets up language classes for indexing characters

# Start and end of word tokens
SOW_token = 0
EOW_token = 1

# Class to set up lemmas and inflected words -- find new name maybe?
class Lang:
    def __init__(self, name):
        self.name = name # S/E
        # Creates an index for char --> index, char count, and index --> char
        #   plus total number of unique chars
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOW", 1: "EOW"}
        self.n_chars = 2  # Count SOW and EOW

    # Adds all characters in word to Lang
    def addWord(self, word, tags=None):
        word = list(word)
        if tags != None:
            word += tags
        for char in word:
            self.addChar(char)

    # Adds char information to/creates indexes and counts for Lang
    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1


# In[ ]:


# Reads in language and setting
# Outputs lemma Lang, inflected word Lang, and pairs plus tags

def readLangs(lang, setting):
    print("Reading lines...")

    # Read the file based on input language and setting (low/med/high) and split into lines
    lines = open('%s_%s.txt' % (lang, setting), encoding='utf-8').        read().strip().split('\n')

    # Split every line into pairs of lemmas/words and tags; normalize
    pairs = [[s for s in l.split('\t')[:2]] for l in lines] # Makes lemma/word pairs
    tags = [l.split('\t')[2].split(';') for l in lines] # Makes list of tags to add to input
    pairs_tags = list(zip(pairs,tags)) # Zips matching lemma/word pairs and tag set

    # Make Lang instances
    lemmas = Lang("lemmas") # S/E
    inflected_words = Lang("inflected words") # S/E

    return lemmas, inflected_words, pairs_tags # S/E


# In[ ]:


# Used to find maximum length for a word to normalize input length
#    runs through all lemma/word pairs to find longest string and returns the length of that string

def findMax(pairs):
    max_length = 0
    for line in pairs:
        if len(line[0][0]) > max_length:
            max_length = len(line[0][0])
        if len(line[0][1]) > max_length:
            max_length =len(line[0][1])
    return max_length


# In[ ]:


# Sets up lemmas, words, lemma/word-pairs and tags pairs, and maximum string length

def prepareData(lang, setting):
    lemmas, inflected_words, pairs_tags = readLangs(lang, setting) # S/E'
    print("Read %s lemma/word pairs" % len(pairs_tags)) # TUOP
    print("Finding maximum string length...") # TUOP
    max_length = findMax(pairs_tags) # S/E
    print("Maximum string length: %s" % max_length) # TUOP
    print("Counting lemmas/words...") # TUOP
    for pairtag in pairs_tags:
        lemmas.addWord(pairtag[0][0],pairtag[1]) # S/E
        inflected_words.addWord(pairtag[0][1]) # S/E
    print("Counted lemmas/words:") # TUOP
    print(lemmas.name, lemmas.n_chars) # TUOP
    print(inflected_words.name, inflected_words.n_chars) # TOUP
    return lemmas, inflected_words, pairs_tags, max_length # S/E


lemmas, inflected_words, pairs_tags, max_length = prepareData('irish', 'low') # S/E
print(random.choice(pairs_tags)) # TUOP


# In[ ]:


# Encoder RNN

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__() # S/E
        self.hidden_size = hidden_size # Sets size of hidden layer

        self.embedding = nn.Embedding(input_size, hidden_size) # Sets embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True) # Sets bidirectional GRU

    def forward(self, input, hidden): # Computes forward propogation
        embedded = self.embedding(input).view(1, 1, -1) # Reshapes tensor
        output = embedded # S/E
        output, hidden = self.gru(output, hidden) # Runs output and hidden through GRU
        return output, hidden # S/E 

    def initHidden(self): # Used to initiate hidden layer dims
        return torch.zeros(2, 1, self.hidden_size, device=device) # Returns torch tensor of zeros in given dims


# In[ ]:


# Decoder RNN with Attention

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__() # S/E
        self.hidden_size = hidden_size # Sets hidden layer size
        self.output_size = output_size # Sets output layer size
        self.dropout_p = dropout_p # Sets dropout rate
        self.max_length = max_length # Sets maximum string length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size) # Sets embedding layer
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length) # Sets attention mechanism
        # Combines attention mecahanims parts?
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size) 
        self.dropout = nn.Dropout(self.dropout_p) # Implements dropout
        self.gru = nn.GRU(self.hidden_size, self.hidden_size) # Sets GRU
        self.out = nn.Linear(self.hidden_size, self.output_size) # Sets linear output function
    
    def forward(self, input, hidden, encoder_outputs): # Forward propogation
        embedded = self.embedding(input).view(1, 1, -1) # Reshapes embedding tensor for hidden layer shape
        embedded = self.dropout(embedded) # Implements dropout of embedded layer

        # Sets attention weights (read more)
        #    takes softmax of linear attention functions from concatenated embedding layer and hidden layer
        #    dim=1 is the dimension along which softmax is applied. why 0 index? (figure out later)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # Matrix multiplication of the attention weights and the encoder outputs
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # ??? Concatenates embedding tensors and attention, but why? and why 0 index?
        output = torch.cat((embedded[0], attn_applied[0]), 1) 
        output = self.attn_combine(output).unsqueeze(0) # Figure all this attention stuff out later

        output = F.relu(output) # Runs output through ReLU function
        output, hidden = self.gru(output, hidden) # Runs output and hidden state through GRU

        # Runs output through linear function, then log softmax; why 0 index and why dim 1?
        output = F.log_softmax(self.out(output[0]), dim=1) 
        return output, hidden, attn_weights # S/E

    def initHidden(self): # Used to initiate hidden layer dims
        return torch.zeros(1, 1, self.hidden_size, device=device) # Returns torch tensor of zeros in given dims


# In[ ]:


# Creates tensors from indexes

def indexesFromWord(Lang, word): # Accesses character indexes of input word from Lang
    return [Lang.char2index[char] for char in word] # S/E

def tensorFromWord(Lang, word): # Creates tensor from input word
    indexes = indexesFromWord(Lang, word) # Gets list of indexes of characters in word
    indexes.append(EOW_token) # Adds end of word token
    # Returns tensor based on character indexes input
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1) # S/E

def indexesFromTags(Tags, tags): # Accesses tag indexes of input tags from Tags
    return [Tags.tag2index[tag] for tag in tags] # S/E

def tensorFromTags(Tags, tags):
    indexes = indexesFromTags(Tags, tags) # Gets list of indexes of characters in word
    # Returns tensor based on character indexes input
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1) # S/E

def tensorsFromPair(pair): # Creates tensor from lemma/word pair
    input_tensor = tensorFromWord(lemmas, pair[0][0]) # Creates input tensor
    target_tensor = tensorFromWord(inflected_words, pair[0][1]) # Creates target output tensor
    return (input_tensor, target_tensor) # S/E


# In[ ]:


# Training loop defintion

teacher_forcing_ratio = 0.5 # Sets probability of teacher forcing occuring

# Defines train function; following is each variable's function, in order:
#    input_tensor = input tensor
#    target_tensor = target tensor
#    encoder = instance of EncoderRNN
#    decoder = instance of AttnDecoderRNN
#    encoder_optimizer, decoder_optimizer = optimization algorithm (in this case SGD)
#    max_length = maximum string length
#    criterion = loss function (in this case negative log loss)
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,          decoder_optimizer, max_length, criterion):
    encoder_hidden = encoder.initHidden() # Returns hidden layer dim for encoder

    encoder_optimizer.zero_grad() # Resets optimizer
    decoder_optimizer.zero_grad() # Resets optimizer

    input_length = input_tensor.size(0) # Input tensor length
    target_length = target_tensor.size(0) # Target tensor length

    # Sets encoder output dims
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size * 2, device=device) 

    loss = 0 # Sets/resets loss to 0

    for ei in range(input_length): # Tensor inputs
        # Calculates encoder output and hidden state based on input tensor and encoder hidden state
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0] # Stores encoder outputs (figure out later why these indexes)
    
    # Initiates decoder input with start of word token
    decoder_input = torch.tensor([[SOW_token]], device=device)
    
    decoder_hidden = encoder_hidden.view(1,1,-1) # Initiates decoder hidden state with final encoder hidden state

    # Randomly decide when to or not to use teacher forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length): # Iterates through target tensors
            # Runs input, decoder hidden state, and encoder outputs through decoder
            #    and sets decoder output, hidden state, and attention to output of decoder
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di]) # Calculates loss
            # Teacher forcing: makes next input the target input instead of guessed output
            decoder_input = target_tensor[di]

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # Runs input, decoder hidden state, and encoder outputs through decoder
            #    and sets decoder output, hidden state, and attention to output of decoder
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1) # ??? Generator? but why topv?
            # "detach from history as input" -- what does this mean/do?
            decoder_input = topi.squeeze().detach()  

            loss += criterion(decoder_output, target_tensor[di]) # Calculates loss
            if decoder_input.item() == EOW_token: # S/E
                break

    loss.backward() # S/E

    encoder_optimizer.step() # S/E
    decoder_optimizer.step() # S/E

    return loss.item() / target_length # S/E


# In[ ]:


# Creates timer functions; Not totally relevant

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[ ]:


# Stuff for plotting; Mess with later

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# In[ ]:


# Defines training iteration function

# Defines training iteration function; explanation of each input vairable:
#    encoder = instance of EncoderRNN
#    decoder = instance of AttnDecoderRNN
#    n_iters = number of iterations
#    print_every=# = print TUOP every # iterations
#    plot_every=# = plot TUOP every # iterations
#    learning_rate=# = S/E
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time() # Starts timer
    plot_losses = [] # ??? Probably for plotting -- deal with later
    print_loss_total = 0  # Reset every print_every -- ??? I think: resets loss total to print
    plot_loss_total = 0  # Reset every plot_every -- ??? I think: resets loss total to plot

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate) # Sets encoder optimizer
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate) # Sets decoder optimizer
    # Sets up training data of tensors from 
    #    randomized selections of lemma/word pairs for the number of iterations
    training_pairs = [tensorsFromPair(random.choice(pairs_tags)) 
                      for i in range(n_iters)]
    criterion = nn.NLLLoss() # Sets loss function

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1] # Selects a training pair
        input_tensor = training_pair[0] # Takes input string's tensor
        target_tensor = training_pair[1] # Take target string's tensor

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, max_length, criterion) # S/E
        print_loss_total += loss # Adds current loss to total loss for printing
        plot_loss_total += loss # Adds current loss to toal loss for plotting

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every # Print avg of loss over iterations
            print_loss_total = 0 # Resets total loss to print
            # TUOP in terms of % complete and time taken, etc
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0: # Plot stuff; check out later
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses) # Shows plot


# In[ ]:


### Evaluation code

# Defines evalutate:
#    encoder, decoder = instances of EncoderRNN, AttnDecoderRNN
#    lemma = word
#    max_length = maximum string length
def evaluate(encoder, decoder, lemma, max_length):
    with torch.no_grad(): # Keeps it from training
        input_tensor = tensorFromWord(lemmas, lemma) # Creates input tensor
        input_length = input_tensor.size()[0] # S/E
        encoder_hidden = encoder.initHidden() # Sets up hidden layer dims

        # Sets up encoder outputs dims and sets them to zeros
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length): # Encoder input
            # Runs input tensor and encoder hidden state through encoder
            #    and sets encoder output and hidden state
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            # Adds encoder output to list of encoder outputs -- details later
            encoder_outputs[ei] += encoder_output[0, 0] 
            

        # Sets encoder input to tensor of start of word token
        decoder_input = torch.tensor([[SOW_token]], device=device)  

        decoder_hidden = encoder_hidden # Sets decoder hidden state to encoder hidden state
        
        decoded_chars = [] # Initiates list of decoded words
        # ??? Sets decoder attentions?
        #    To tensor of zeros of maximum string length as dimensions? Why those dims?
        #    Figure that out later
        decoder_attentions = torch.zeros(max_length, max_length) 

        for di in range(max_length): # Decoder input
            # Runs input tensor and decoder hidden state through decoder
            #    and sets decoder output and hidden state
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data # ??? .data? Probably the generator
            topv, topi = decoder_output.data.topk(1) # ??? topv, topi? .data.topk?
            if topi.item() == EOW_token: # Breaks if top word is end of word token(? what is topi?)
                decoded_chars.append('<EOW>')
                break
            else:
                # Appends topi(?) word to decoded word list
                decoded_chars.append(inflected_words.index2char[topi.item()]) 
                
            decoder_input = topi.squeeze().detach() # Details later

        return decoded_chars, decoder_attentions[:di + 1] # Returns decoded word and attentions


# In[ ]:


# Randomly evaluate words

def evaluateRandomly(encoder, decoder, pairs_tags, n=10): # n=# = number of samples to evaluate
    for i in range(n):
        pair = random.choice(pairs_tags)
        print('>', pair[0][0])
        print('=', pair[0][1])
        output_chars, attentions = evaluate(encoder, decoder, pair[0][0], pair[1], max_length)
        output_word = ''.join(output_chars)
        print('<', output_word)
        print('')


# In[ ]:


hidden_size = 256 # Hidden layer size
# Initiates instance of EncoderRNN with input size of number of unique chars? and hidden size as above
encoder1 = EncoderRNN(lemmas.n_chars, hidden_size).to(device) 
# Initiates instance of AttnDecoderRNN with hidden size as above, output size of number of unique chars?
#    the maximum string length, and the dropout rate
attn_decoder1 = AttnDecoderRNN(hidden_size * 2, inflected_words.n_chars, max_length, dropout_p=0.1).to(device)

# Executes program
trainIters(encoder1, attn_decoder1, 75000, print_every=5000)


# In[ ]:


evaluateRandomly(encoder1, attn_decoder1, pairs_tags) # Evaluates random samples


# In[ ]:


# Implements evaluation of system

output_chars, attentions = evaluate(
    encoder1, attn_decoder1, "amhran", max_length) # Outputs word and attentions from system given input word
plt.matshow(attentions.numpy()) # Plots attentions


# In[ ]:


# Creates plots of attentions and evaluations of system output
# DO THIS LATER

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("INSERT WORD HERE")

evaluateAndShowAttention("INSERT WORD HERE")

evaluateAndShowAttention("INSERT WORD HERE")

evaluateAndShowAttention("INSERT WORD HERE")


# In[ ]:


dev_lemmas, dev_inflected_words, dev_pairs_tags, dev_max_length = prepareData('irish', 'dev') # S/E
print(random.choice(pairs_tags)) # TUOP


# In[ ]:


evaluateRandomly(encoder1, attn_decoder1,dev_pairs_tags) # Evaluates random samples

