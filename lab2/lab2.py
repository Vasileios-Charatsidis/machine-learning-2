
# coding: utf-8

# # Lab 2: Inference in Graphical Models
# 
# ### Machine Learning 2, 2016
# 
# * The lab exercises should be made in groups of two people.
# * The deadline is Sunday, April 24, 23:59.
# * Assignment should be sent to t.s.cohen at uva dot nl (Taco Cohen). The subject line of your email should be "[ML2_2016] lab#_lastname1\_lastname2". 
# * Put your and your teammate's names in the body of the email
# * Attach the .IPYNB (IPython Notebook) file containing your code and answers. Naming of the file follows the same rule as the subject line.
# 
# Notes on implementation:
# 
# * You should write your code and answers in an IPython Notebook: http://ipython.org/notebook.html. If you have problems, please contact us.
# * Among the first lines of your notebook should be "%pylab inline". This imports all required modules, and your plots will appear inline.
# * NOTE: test your code and make sure we can run your notebook / scripts!

# ### Introduction
# In this assignment, we will implement the sum-product and max-sum algorithms for factor graphs over discrete variables. The relevant theory is covered in chapter 8 of Bishop's PRML book, in particular section 8.4. Read this chapter carefuly before continuing!
# 
# We will first implement sum-product and max-sum and apply it to a simple poly-tree structured factor graph for medical diagnosis. Then, we will implement a loopy version of the algorithms and use it for image denoising.
# 
# For this assignment we recommended you stick to numpy ndarrays (constructed with np.array, np.zeros, np.ones, etc.) as opposed to numpy matrices, because arrays can store n-dimensional arrays whereas matrices only work for 2d arrays. We need n-dimensional arrays in order to store conditional distributions with more than 1 conditioning variable. If you want to perform matrix multiplication on arrays, use the np.dot function; all infix operators including *, +, -, work element-wise on arrays.

# ## Part 1: The sum-product algorithm
# 
# We will implement a datastructure to store a factor graph and to facilitate computations on this graph. Recall that a factor graph consists of two types of nodes, factors and variables. Below you will find some classes for these node types to get you started. Carefully inspect this code and make sure you understand what it does; you will have to build on it later.

# In[1]:

get_ipython().magic(u'pylab gtk')
import logging
import sys
from types import MethodType

LOG_LEVEL = logging.WARNING
LOG_FORMAT = '%(name)s - %(message)s'
logging.basicConfig(format=LOG_FORMAT)
# Make root logger print in notebook
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)
# root_logger.handlers[0].stream = sys.stdout


# In[2]:

class Node(object):
    """
    Base-class for Nodes in a factor graph. Only instantiate sub-classes of Node.
    """
    def __init__(self, name):
        # A name for this Node, for printing purposes
        self.name = name
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVEL)
        
        # Neighbours in the graph, identified with their index in this list.
        # i.e. self.neighbours contains neighbour 0 through len(self.neighbours) - 1.
        self.neighbours = []
        
        # Reset the node-state (not the graph topology)
        self.reset()
    
    def reset(self):
        # Incoming messages; a dictionary mapping neighbours to messages.
        # That is, it maps  Node -> np.ndarray.
        self.in_msgs = {}
        
        # A set of neighbours for which this node has pending messages.
        # We use a python set object so we don't have to worry about duplicates.
        self.pending = set([])

    def add_neighbour(self, nb):
        self.neighbours.append(nb)

    def send_sp_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_sp_msg not implemented in base-class Node')
   
    def send_ms_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_ms_msg not implemented in base-class Node')
    
    def receive_msg(self, other, msg):
        # Store the incomming message, replacing previous messages from the same node
        self.in_msgs[other] = msg

        # TODO: add pending messages
        # self.pending.update(...)
    
    def __str__(self):
        # This is printed when using 'print node_instance'
        return self.name


class Variable(Node):
    def __init__(self, name, num_states):
        """
        Variable node constructor.
        Args:
            name: a name string for this node. Used for printing. 
            num_states: the number of states this variable can take.
            Allowable states run from 0 through (num_states - 1).
            For example, for a binary variable num_states=2,
            and the allowable states are 0, 1.
        """
        self.num_states = num_states
        
        # Call the base-class constructor
        super(Variable, self).__init__(name)
    
    def set_observed(self, observed_state):
        """
        Set this variable to an observed state.
        Args:
            observed_state: an integer value in [0, self.num_states - 1].
        """
        # Observed state is represented as a 1-of-N variable
        # Could be 0.0 for sum-product, but log(0.0) = -inf so a tiny value is preferable for max-sum
        self.observed_state[:] = 0.000001
        self.observed_state[observed_state] = 1.0
        self.is_observed = True
        
    def set_latent(self):
        """
        Erase an observed state for this variable and consider it latent again.
        """
        # No state is preferred, so set all entries of observed_state to 1.0
        # Using this representation we need not differentiate between observed and latent
        # variables when sending messages.
        self.observed_state[:] = 1.0
        self.is_observed = False
        
    def reset(self):
        super(Variable, self).reset()
        self.observed_state = np.ones(self.num_states)
        self.is_observed = False
        
    def marginal(self, Z=None):
        """
        Compute the marginal distribution of this Variable.
        It is assumed that message passing has completed when this function is called.
        Args:
            Z: an optional normalization constant can be passed in. If None is passed, Z is computed.
        Returns: marginal, Z. The first is a numpy array containing the normalized marginal distribution.
         Z is either equal to the input Z, or computed in this function (if Z=None was passed).
        """
        # TODO: compute marginal
        return None, Z
    
    def send_sp_msg(self, other):
        # TODO: implement Variable -> Factor message for sum-product
        pass
   
    def send_ms_msg(self, other):
        # TODO: implement Variable -> Factor message for max-sum
        pass

class Factor(Node):
    def __init__(self, name, f, neighbours):
        """
        Factor node constructor.
        Args:
            name: a name string for this node. Used for printing
            f: a numpy.ndarray with N axes, where N is the number of neighbours.
               That is, the axes of f correspond to variables, and the index along that axes corresponds to a value of that variable.
               Each axis of the array should have as many entries as the corresponding neighbour variable has states.
            neighbours: a list of neighbouring Variables. Bi-directional connections are created.
        """
        # Call the base-class constructor
        super(Factor, self).__init__(name)

        assert len(neighbours) == f.ndim, 'Factor function f should accept as many arguments as this Factor node has neighbours'
        
        for nb_ind in range(len(neighbours)):
            nb = neighbours[nb_ind]
            assert f.shape[nb_ind] == nb.num_states, 'The range of the factor function f is invalid for input %i %s' % (nb_ind, nb.name)
            self.add_neighbour(nb)
            nb.add_neighbour(self)

        self.f = f
        
    def send_sp_msg(self, other):
        # TODO: implement Factor -> Variable message for sum-product
        pass
   
    def send_ms_msg(self, other):
        # TODO: implement Factor -> Variable message for max-sum
        pass


# ### 1.1 Instantiate network (10 points)
# Convert the directed graphical model ("Bayesian Network") shown below to a factor graph. Instantiate this graph by creating Variable and Factor instances and linking them according to the graph structure. 
# To instantiate the factor graph, first create the Variable nodes and then create Factor nodes, passing a list of neighbour Variables to each Factor.
# Use the following prior and conditional probabilities.
# 
# $$
# p(\verb+Influenza+) = 0.05 \\\\
# p(\verb+Smokes+) = 0.2 \\\\
# $$
# 
# $$
# p(\verb+SoreThroat+ = 1 | \verb+Influenza+ = 1) = 0.3 \\\\
# p(\verb+SoreThroat+ = 1 | \verb+Influenza+ = 0) = 0.001 \\\\
# p(\verb+Fever+ = 1| \verb+Influenza+ = 1) = 0.9 \\\\
# p(\verb+Fever+ = 1| \verb+Influenza+ = 0) = 0.05 \\\\
# p(\verb+Bronchitis+ = 1 | \verb+Influenza+ = 1, \verb+Smokes+ = 1) = 0.99 \\\\
# p(\verb+Bronchitis+ = 1 | \verb+Influenza+ = 1, \verb+Smokes+ = 0) = 0.9 \\\\
# p(\verb+Bronchitis+ = 1 | \verb+Influenza+ = 0, \verb+Smokes+ = 1) = 0.7 \\\\
# p(\verb+Bronchitis+ = 1 | \verb+Influenza+ = 0, \verb+Smokes+ = 0) = 0.0001 \\\\
# p(\verb+Coughing+ = 1| \verb+Bronchitis+ = 1) = 0.8 \\\\
# p(\verb+Coughing+ = 1| \verb+Bronchitis+ = 0) = 0.07 \\\\
# p(\verb+Wheezing+ = 1| \verb+Bronchitis+ = 1) = 0.6 \\\\
# p(\verb+Wheezing+ = 1| \verb+Bronchitis+ = 0) = 0.001 \\\\
# $$

# In[3]:

from IPython.core.display import Image 
Image(filename='bn.png') 


# In[4]:

# Create variable nodes
influenza = Variable('Influenza', 2)
smokes = Variable('Smokes', 2)
sore_throat = Variable('SoreThroat', 2)
fever = Variable('Fever', 2)
bronchitis = Variable('Bronchitis', 2)
coughing = Variable('Coughing', 2)
wheezing = Variable('Wheezing', 2)

# Create factor nodes
influenza_prior = Factor('Influenza prior', np.array([1. - .05, .05]), [influenza])
smokes_prior = Factor('Smokes prior', np.array([1. - .2, .2]), [smokes])
st_given_flu = Factor('SoreThroat given Influenza', np.array([
            [1. - .001, 1. - .3],
            [.001, .001]
        ]), [sore_throat, influenza])
fever_given_flu = Factor('Fever given Influenza', np.array([
            [1. - .05, 1. - .9],
            [.05, .9]
        ]), [fever, influenza])
bronchitis_given_flu_smokes = Factor('Bronchitis given Inf. and Smokes', np.array([
            [
                [1. - .0001, 1. - .7],
                [1. - .9, 1. - .99]
            ], [
                [.0001, .7],
                [.9, .99]
            ]
        ]), [bronchitis, influenza, smokes])
coughing_given_bron = Factor('Coughing given Bronchitis', np.array([
            [1. - .07, 1. - .8],
            [.07, .8]
        ]), [coughing, bronchitis])
wheezing_given_bron = Factor('Wheezing given Bronchitis', np.array([
            [1. - .001, 1. - .6],
            [.001, .6]
        ]), [wheezing, bronchitis])


all_nodes = [influenza, smokes, sore_throat, coughing, wheezing, bronchitis, fever,
            influenza_prior, smokes_prior, st_given_flu, coughing_given_bron, wheezing_given_bron,
            bronchitis_given_flu_smokes, fever_given_flu]


# ### 1.2 Factor to variable messages (20 points)
# Write a method `send_sp_msg(self, other)` for the Factor class, that checks if all the information required to pass a message to Variable `other` is present, computes the message and sends it to `other`. "Sending" here simply means calling the `receive_msg` function of the receiving node (we will implement this later). The message itself should be represented as a numpy array (np.array) whose length is equal to the number of states of the variable.
# 
# An elegant and efficient solution can be obtained using the n-way outer product of vectors. This product takes n vectors $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}$ and computes a $n$-dimensional tensor (ndarray) whose element $i_0,i_1,...,i_n$ is given by $\prod_j \mathbf{x}^{(j)}_{i_j}$. In python, this is realized as `np.multiply.reduce(np.ix_(*vectors))` for a python list `vectors` of 1D numpy arrays. Try to figure out how this statement works -- it contains some useful functional programming techniques. Another function that you may find useful in computing the message is `np.tensordot`.

# In[5]:

def node_check_send_info(self, other):
    assert other in self.neighbours,         'Trying to send message to not neighbouring node {} -> {}'.format(self.name, other.name)
    # Check if all information required is present
    # that is check if have received messages from all neighboring
    # variables except for 'other'
    received_from = self.in_msgs.keys()
    other_neighbours = [n for n in self.neighbours if n != other]
    all_received = reduce(lambda acc, n: acc and n in received_from, other_neighbours, True)
    assert all_received, 'Node {} has not all needed information'.format(self.name)


# Bound function as Node class method
Node.check_send_info = MethodType(node_check_send_info, None, Node)


# In[6]:

def factor_send_sp_msg(self, other):
    self.logger.info('Sending message to %s', other)
    # Check if all information required is present
    self.check_send_info(other)
    
    # Compute the message
    if len(self.neighbours) == 1:
        msg = self.f
    else:
        other_neighbours = [n for n in self.neighbours if n != other]
        vectors = [self.in_msgs[node] for node in other_neighbours]
        self.logger.debug('Factor vectors\n%s', vectors)
        multiplication = np.multiply.reduce(np.ix_(*vectors))
        other_i = self.neighbours.index(other)
        f_axes = filter(lambda i: not i == other_i, range(len(self.neighbours)))
        msg = np.tensordot(self.f, multiplication, axes=(f_axes, range(multiplication.ndim)))

    # Send message to other
    other.receive_msg(self, msg)
    self.pending.discard(other)


# Bound function as Factor class method
Factor.send_sp_msg = MethodType(factor_send_sp_msg, None, Factor)


# ### 1.3 Variable to factor messages (10 points)
# 
# Write a method `send_sp_message(self, other)` for the Variable class, that checks if all the information required to pass a message to Variable var is present, computes the message and sends it to factor.

# In[7]:

def variable_send_sp_msg(self, other):
    self.logger.info('Sending message to %s', other)
    # Check if all information required is present
    self.check_send_info(other)

    # Compute the message
    if len(self.neighbours) == 1:
        msg = np.ones(self.num_states)
    else:
        other_neighbours = [n for n in self.neighbours if n != other]
        vectors = [self.in_msgs[node] for node in other_neighbours]
        self.logger.debug('Variable vectors\n%s', vectors)
        msg = np.multiply.reduce(vectors)
        
    # Send the message
    other.receive_msg(self, msg)
    self.pending.discard(other)


# Bound function as Variable class method
Variable.send_sp_msg = MethodType(variable_send_sp_msg, None, Variable)


# ### 1.4 Compute marginal (10 points)
# Later in this assignment, we will implement message passing schemes to do inference. Once the message passing has completed, we will want to compute local marginals for each variable.
# Write the method `marginal` for the Variable class, that computes a marginal distribution over that node.

# In[8]:

def marginal(self, Z=None):
    vectors = [self.in_msgs[node] for node in self.neighbours]
    unnormalized = np.multiply.reduce(vectors)
    if not Z:
        Z = unnormalized.sum()
    marginal = unnormalized / Z
    return marginal, Z


# Bound function as Variable class method
Variable.marginal = MethodType(marginal, None, Variable)


# ### 1.5 Receiving messages (10 points)
# In order to implement the loopy and non-loopy message passing algorithms, we need some way to determine which nodes are ready to send messages to which neighbours. To do this in a way that works for both loopy and non-loopy algorithms, we make use of the concept of "pending messages", which is explained in Bishop (8.4.7): 
# "we will say that a (variable or factor)
# node a has a message pending on its link to a node b if node a has received any
# message on any of its other links since the last time it send (sic) a message to b. Thus,
# when a node receives a message on one of its links, this creates pending messages
# on all of its other links."
# 
# Keep in mind that for the non-loopy algorithm, nodes may not have received any messages on some or all of their links. Therefore, before we say node a has a pending message for node b, we must check that node a has received all messages needed to compute the message that is to be sent to b.
# 
# Modify the function `receive_msg`, so that it updates the self.pending variable as described above. The member self.pending is a set that is to be filled with Nodes to which self has pending messages. Modify the `send_msg` functions to remove pending messages as they are sent.

# In[9]:

def receive_msg(self, other, msg):
    # Store the incomming message, replacing previous messages from the same node
    self.in_msgs[other] = msg
    self.logger.debug('Received message from %s: %s', other, msg)

    all_other_nodes = [node for node in self.neighbours if node != other]
    received_from = set(self.in_msgs.keys())
    to_append_nodes = []
    for node in all_other_nodes:
        other_nodes = set([n for n in self.neighbours if n != node])
        if other_nodes <= received_from:
            to_append_nodes.append(node)

    self.pending.update(to_append_nodes)
    # self.pending.update(set(self.neighbours) - set([other]))
    
    for p in self.pending: self.logger.debug('Pending: %s', p)


# Bound function as Node class method
Node.receive_msg = MethodType(receive_msg, None, Node)


# ### 1.6 Inference Engine (10 points)
# Write a function `sum_product(node_list)` that runs the sum-product message passing algorithm on a tree-structured factor graph with given nodes. The input parameter `node_list` is a list of all Node instances in the graph, which is assumed to be ordered correctly. That is, the list starts with a leaf node, which can always send a message. Subsequent nodes in `node_list` should be capable of sending a message when the pending messages of preceding nodes in the list have been sent. The sum-product algorithm then proceeds by passing over the list from beginning to end, sending all pending messages at the nodes it encounters. Then, in reverse order, the algorithm traverses the list again and again sends all pending messages at each node as it is encountered. For this to work, you must initialize pending messages for all the leaf nodes, e.g. `influenza_prior.pending.add(influenza)`, where `influenza_prior` is a Factor node corresponding the the prior, `influenza` is a Variable node and the only connection of `influenza_prior` goes to `influenza`.
# 
# 
# 

# In[10]:

def sum_product(node_list):
    # Performs a single pass in one direction
    def do_pass(n_list):
        for i in range(len(n_list)):
            node = n_list[i]
            root_logger.debug('Pass %d %s', i, node)
            pending_nodes = list(node.pending)
            for pending in pending_nodes:
                node.send_sp_msg(pending)
            root_logger.debug('')
    
    # Forward pass
    do_pass(node_list)
    # Backward pass
    do_pass(list(reversed(node_list)))


# In[11]:

def init_graph(observed=[]):
    # Reset all nodes
    for node in all_nodes: node.reset()
    
    # Setup leaf nodes
    influenza_prior.pending.add(influenza)
    smokes_prior.pending.add(smokes)
    sore_throat.pending.add(st_given_flu)
    fever.pending.add(fever_given_flu)
    coughing.pending.add(coughing_given_bron)
    wheezing.pending.add(wheezing_given_bron)

    # Setup graph ordering
    return [influenza_prior, smokes_prior, coughing,
            wheezing, sore_throat, fever, st_given_flu,
            fever_given_flu, coughing_given_bron, wheezing_given_bron,
            smokes, influenza, bronchitis, bronchitis_given_flu_smokes]

graph = init_graph()
# Start sum-product algorithm
sum_product(graph)

st_marginal, _ = sore_throat.marginal()
print 'p(sore_throat=0) =', st_marginal[0]
print 'p(sore_throat=1) =', st_marginal[1]


# ### 1.7 Observed variables and probabilistic queries (15 points)
# We will now use the inference engine to answer probabilistic queries. That is, we will set certain variables to observed values, and obtain the marginals over latent variables. We have already provided functions `set_observed` and `set_latent` that manage a member of Variable called `observed_state`. Modify the `Variable.send_msg` and `Variable.marginal` routines that you wrote before, to use `observed_state` so as to get the required marginals when some nodes are observed.

# In[12]:

def variable_send_sp_msg_obs(self, other):
    self.logger.info('Sending message to %s', other)
    
    # Compute the message
    if len(self.neighbours) == 1 or self.is_observed:
        msg = self.observed_state
    else:
        # Check if all information required is present
        self.check_send_info(other)
        other_neighbours = [n for n in self.neighbours if n != other]
        vectors = [self.in_msgs[node] for node in other_neighbours]
        self.logger.debug('Variable vectors\n%s', vectors)
        msg = np.multiply.reduce(vectors)
        
    # Send the message
    other.receive_msg(self, msg)
    self.pending.discard(other)


def marginal_obs(self, Z=None):
    if self.is_observed:
        unnormalized = self.observed_state
    else:
        vectors = [self.in_msgs[node] for node in self.neighbours]
        unnormalized = np.multiply.reduce(vectors)
    if not Z:
        Z = unnormalized.sum()
    marginal = unnormalized / Z
    return marginal, Z


# Bound functions as Variable class method
Variable.marginal = MethodType(marginal_obs, None, Variable)
Variable.send_sp_msg = MethodType(variable_send_sp_msg_obs, None, Variable)


# In[13]:

graph = init_graph()
influenza.set_observed(1)
smokes.set_observed(1)
bronchitis.set_observed(1)

sum_product(graph)

coughing_marginal, _ = coughing.marginal()
print 'p(coughing=0|influenza=1,smokes=1,bronchitis=1) =', coughing_marginal[0]
print 'p(coughing=1|influenza=1,smokes=1,bronchitis=1) =', coughing_marginal[1]

st_marginal, _ = sore_throat.marginal()
print 'p(sore_throat=0|influenza=1,smokes=1,bronchitis=1) =', st_marginal[0]
print 'p(sore_throat=1|influenza=1,smokes=1,bronchitis=1) =', st_marginal[1]


# In[14]:

graph = init_graph()
influenza.set_observed(1)
smokes.set_observed(1)
bronchitis.set_observed(0)

sum_product(graph)

coughing_marginal, _ = coughing.marginal()
print 'p(coughing=0|influenza=1,smokes=1,bronchitis=0) =', coughing_marginal[0]
print 'p(coughing=1|influenza=1,smokes=1,bronchitis=0) =', coughing_marginal[1]

st_marginal, _ = sore_throat.marginal()
print 'p(sore_throat=0|influenza=1,smokes=1,bronchitis=0) =', st_marginal[0]
print 'p(sore_throat=1|influenza=1,smokes=1,bronchitis=0) =', st_marginal[1]


# In[15]:

graph = init_graph()
influenza.set_observed(1)
smokes.set_observed(1)

sum_product(graph)

coughing_marginal, _ = coughing.marginal()
print 'p(coughing=0|influenza=1,smokes=1) =', coughing_marginal[0]
print 'p(coughing=1|influenza=1,smokes=1) =', coughing_marginal[1]

st_marginal, _ = sore_throat.marginal()
print 'p(sore_throat=0|influenza=1,smokes=1) =', st_marginal[0]
print 'p(sore_throat=1|influenza=1,smokes=1) =', st_marginal[1]


# In[16]:

graph = init_graph()
influenza.set_observed(1)
smokes.set_observed(0)

sum_product(graph)

coughing_marginal, _ = coughing.marginal()
print 'p(coughing=0|influenza=1,smokes=0) =', coughing_marginal[0]
print 'p(coughing=1|influenza=1,smokes=0) =', coughing_marginal[1]

st_marginal, _ = sore_throat.marginal()
print 'p(sore_throat=0|influenza=1,smokes=0) =', st_marginal[0]
print 'p(sore_throat=1|influenza=1,smokes=0) =', st_marginal[1]


# ### 1.8 Sum-product and MAP states (5 points)
# A maximum a posteriori state (MAP-state) is an assignment of all latent variables that maximizes the probability of latent variables given observed variables:
# $$
# \mathbf{x}_{\verb+MAP+} = \arg\max _{\mathbf{x}} p(\mathbf{x} | \mathbf{y})
# $$
# Could we use the sum-product algorithm to obtain a MAP state? If yes, how? If no, why not?
# 

# 

# ## Part 2: The max-sum algorithm
# Next, we implement the max-sum algorithm as described in section 8.4.5 of Bishop.

# ### 2.1 Factor to variable messages (10 points)
# Implement the function `Factor.send_ms_msg` that sends Factor -> Variable messages for the max-sum algorithm. It is analogous to the `Factor.send_sp_msg` function you implemented before.

# In[17]:

def factor_send_ms_msg(self, other):
    self.logger.info('Sending message to %s', other)
    self.check_send_info(other)
    
    # Compute the message
    if len(self.neighbours) == 1:
        msg = np.log(self.f)
    else:
        other_neighbours = [n for n in self.neighbours if n != other]
        mus = 0
        dims = [n.num_states for n in other_neighbours]
        for i, node in enumerate(other_neighbours):
            dims_i = dims[:]
            dims_i.pop(i)
            dims_i.append(1)
            mu = np.tile(self.in_msgs[node], dims_i)
            mus += mu
        other_i = self.neighbours.index(other)
        f_axes = filter(lambda i: i != other_i, range(len(self.neighbours)))
        msg = np.amax(np.log(self.f) + mus, axis=tuple(f_axes))

    # Send message to other
    other.receive_msg(self, msg)
    self.pending.discard(other)


# Bound function as Factor class method
Factor.send_ms_msg = MethodType(factor_send_ms_msg, None, Factor)


# ### 2.2 Variable to factor messages (10 points)
# Implement the `Variable.send_ms_msg` function that sends Variable -> Factor messages for the max-sum algorithm.

# In[18]:

def variable_send_ms_msg(self, other):
    self.logger.info('Sending message to %s', other)
    
    # Compute the message
    if len(self.neighbours) == 1 or self.is_observed:
        msg = np.log(self.observed_state)
    else:
        # Check if all information required is present
        self.check_send_info(other)
        msg = 0
        for node in [n for n in self.neighbours if n != other]:
            msg += self.in_msgs[node]
        
    # Send the message
    other.receive_msg(self, msg)
    self.pending.discard(other)


Variable.send_ms_msg = MethodType(variable_send_ms_msg, None, Variable)


# ### 2.3 Find a MAP state (10 points)
# 
# Using the same message passing schedule we used for sum-product, implement the max-sum algorithm. For simplicity, we will ignore issues relating to non-unique maxima. So there is no need to implement backtracking; the MAP state is obtained by a per-node maximization (eq. 8.98 in Bishop). Make sure your algorithm works with both latent and observed variables.

# In[19]:

def map_state(self):
    return np.argmax(np.sum(self.in_msgs.values(), axis=0))

Variable.map_state = MethodType(map_state, None, Variable)


def max_sum(node_list):
    # Performs a single pass in one direction
    def do_pass(n_list):
        for i in range(len(n_list)):
            node = n_list[i]
            root_logger.debug('Pass %d %s', i, node)
            pending_nodes = list(node.pending)
            for pending in pending_nodes:
                node.send_ms_msg(pending)
            root_logger.debug('')
    
    # Forward pass
    do_pass(node_list)
    # Backward pass
    do_pass(list(reversed(node_list)))
    
    return {var.name: var.map_state() for var in node_list if isinstance(var, Variable)}


# In[20]:

graph = init_graph()
max_sum(graph)


# In[21]:

graph = init_graph()
influenza.set_observed(1)
smokes.set_observed(1)

max_sum(graph)


# In[22]:

graph = init_graph()
fever.set_observed(1)
coughing.set_observed(1)
sore_throat.set_observed(1)
wheezing.set_observed(0)

max_sum(graph)


# ## Part 3: Image Denoising and Loopy BP
# 
# Next, we will use a loopy version of max-sum to perform denoising on a binary image. The model itself is discussed in Bishop 8.3.3, but we will use loopy max-sum instead of Iterative Conditional Modes as Bishop does.
# 
# The following code creates some toy data. `im` is a quite large binary image, `test_im` is a smaller synthetic binary image. Noisy versions are also provided.

# In[23]:

from pylab import imread, gray
# Load the image and binarize
im = np.mean(imread('dalmatian1.png'), axis=2) > 0.5
imshow(im)
gray()

# Add some noise
noise = np.random.rand(*im.shape) > 0.9
noise_im = np.logical_xor(noise, im)
figure()
imshow(noise_im)

test_im = np.zeros((10,10))
test_im[5:8, 3:8] = 1.0
test_im[5,5] = 1.0
figure()
imshow(test_im)

# Add some noise
noise = np.random.rand(*test_im.shape) > 0.9
noise_test_im = np.logical_xor(noise, test_im)
figure()
imshow(noise_test_im)


# ### 3.1 Construct factor graph (10 points)
# Convert the Markov Random Field (Bishop, fig. 8.31) to a factor graph and instantiate it.

# In[24]:

class DenoisingGraph(object):
    def __init__(self, im, noise_prior=.9, correlation_prior=.7):
        # Setup logger
        self.logger = logging.getLogger('DenoisingGraph')
        self.logger.setLevel(LOG_LEVEL)
        self.im = im
        self._init_graph(noise_prior, correlation_prior)
        
    def _init_graph(self, noise_prior, correlation_prior):
        height = self.im.shape[0]
        width = self.im.shape[1]
        noise_prior_f = np.array([
                [1. - noise_prior, noise_prior],
                [1. - noise_prior, noise_prior]])
        correlation_prior_f = np.array([
                [1. - correlation_prior, correlation_prior],
                [1. - correlation_prior, correlation_prior]])

        # Create variables for noisy and denoised pixels
        # and factors for noise-denoise variables
        self.noisy_vars = []
        self.denoised_vars = []
        self.nd_factors = set([])
        for h in range(height):
            noisy_row = []
            denoised_row = []
            for w in range(width):
                noisy_px = Variable('y-{}-{}'.format(h, w), 2)
                # Set observed state to noisy pixel value
                noisy_px.set_observed(self.im[h,w])
                noisy_row.append(noisy_px)
                denoised_px = Variable('x-{}-{}'.format(h, w), 2)
                denoised_row.append(denoised_px)
                # Create factor node
                neighbours = [noisy_px, denoised_px]
                factor = Factor('ndf-{}-{}'.format(h, w), noise_prior_f, neighbours)
                self.nd_factors.add(factor)
            self.noisy_vars.append(noisy_row)
            self.denoised_vars.append(denoised_row)

        # Create factors for neighbouring nodes
        self.neigh_factors = []
        for h in range(height-1):
            frow = []
            for w in range(width-1):
                neighbours1 = [self.denoised_vars[h][w+1], self.denoised_vars[h+1][w]]
                factor1 = Factor('neigh-tb-{}-{}'.format(h, w), correlation_prior_f, neighbours1)
                neighbours2 = [self.denoised_vars[h][w], self.denoised_vars[h+1][w+1]]
                factor2 = Factor('neigh-lr-{}-{}'.format(h, w), correlation_prior_f, neighbours2)
                frow.append([factor1, factor2])
            self.neigh_factors.append(frow)


# In[25]:

denoising_graph_test = DenoisingGraph(noise_test_im)


# ### 3.2 Loopy max-sum (10 points)
# Implement the loopy max-sum algorithm, by passing messages from randomly chosen nodes iteratively until no more pending messages are created or a maximum number of iterations is reached. 
# 
# Think of a good way to initialize the messages in the graph.

# In[26]:

def loopy_max_sum(self, max_iterations):
    # Setup leaf nodes
    self.logger.info('Setting up leaf nodes')
    for nrow in self.noisy_vars:
        for n_px in nrow:
            n_px.pending.add(n_px.neighbours[0])

    def send_pending(node):
        # print 'Sending pending', len(node.pending)
        pending_cpy = node.pending.copy()
        for pending in pending_cpy:
            node.send_ms_msg(pending)
            
    # Perform message passing
    self.logger.info('Initial messages from observed nodes')
    for nrow in self.noisy_vars:
        for n_px in nrow:
            send_pending(n_px)
    self.logger.info('Initial messages from noisy-denoised factors')
    for nd_factor in self.nd_factors:
        send_pending(nd_factor)
    
    for iteration in range(max_iterations):
        self.logger.info('Message-passing iteration %d', iteration)
        if iteration % 2 == 0:
            for drow in self.denoised_vars:
                for d_px in drow:
                    send_pending(d_px)
            for frow in self.neigh_factors:
                for fs in frow:
                    for f in fs: send_pending(f)
        else:
            for frow in self.neigh_factors:
                for fs in frow:
                    for f in fs: send_pending(f)
            for drow in self.denoised_vars:
                for d_px in drow:
                    send_pending(d_px)
    
    self.logger.info('Final messages from noisy-denoised factors')
    for nd_factor in self.nd_factors:
        send_pending(nd_factor)
    self.logger.info('Final messages from observed nodes')
    for nrow in self.noisy_vars:
        for n_px in nrow:
            send_pending(n_px)
    

def denoise_im(self, max_iterations=10, show=True):
    self._max_sum(max_iterations)
    
    clean_im = []
    for xrow in self.denoised_vars:
        clean_row = []
        for d_px in xrow:
            clean_row.append(d_px.map_state())
        clean_im.append(clean_row)
        
    if show:
        plt.figure()
        plt.imshow(np.array(clean_im))
    return clean_im
    
    
DenoisingGraph._max_sum = MethodType(loopy_max_sum, None, DenoisingGraph)
DenoisingGraph.denoise_im = MethodType(denoise_im, None, DenoisingGraph)


# In[27]:

denoised_test = denoising_graph_test.denoise_im(show=False)

plt.figure(figsize=(12,6))
f = plt.subplot(1,3,1)
f.set_title('Original')
imshow(test_im)
f = plt.subplot(1,3,2)
f.set_title('Noisy')
imshow(noise_test_im)
f = plt.subplot(1,3,3)
f.set_title('Denoised')
imshow(denoised_test)


# In[28]:

denoising_graph = DenoisingGraph(noise_im)
denoised_im = denoising_graph.denoise_im(show=False)

plt.figure(figsize=(12,6))
f = plt.subplot(1,3,1)
f.set_title('Original')
imshow(im)
f = plt.subplot(1,3,2)
f.set_title('Noisy')
imshow(noise_im)
f = plt.subplot(1,3,3)
f.set_title('Denoised')
imshow(denoised_im)


# In[29]:

print np.sum(im == noise_im), im.size
print np.sum(im == denoised_im), im.size

