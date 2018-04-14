'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import numpy as np
import networkx as nx
import Embedding_models.Node2Vec.node2vec as n2v
from gensim.models import Word2Vec
import pandas as pd

def read_graph(args):
    '''
    Reads the input network in networkx.
    '''
    graph_edges = pd.read_csv(args.input, index_col = None).values.tolist()

    if args.weighted:
        G = nx.DiGraph() 
        for elem in graph_edges:
            G.add_edge(elem[0],elem[2],label=elem[1])

        # G = nx.read_edgelist(args.input, nodetype=str, edgetype=str, data=(('weight',str),), create_using=nx.DiGraph())
    else:
        for elem in graph_edges:
            G.add_edge(elem[0],elem[2])
        # G = nx.read_edgelist(args.input, nodetype=str, create_using=nx.DiGraph())

    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G

def learn_embeddings(walks, args):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    # walks_node = [str(walk) for walk in walks]
    # walks = [w for walk in walks for w in walk]
    walks_str = []
    for walk in walks:
        sentence = []
        for word in walk:
            sentence.append(str(word))
        walks_str.append(sentence)

    model = Word2Vec(walks_str, size = args.dimensions, window = args.window_size, min_count = 1, sg = 1, workers = args.workers, iter = args.iter, alpha = args.alpha)
    model.wv.save_word2vec_format(args.output+"node2vec"+ str(args.dimensions) +".emb")
    
    return

def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    print("node2vec started")
    nx_G = read_graph(args)
    G = n2v.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks, args)

