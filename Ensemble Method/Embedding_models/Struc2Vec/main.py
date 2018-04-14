#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse, logging
import numpy as np
import Embedding_models.Struc2Vec.struc2vec as struc2vec 
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time
import pandas as pd
import networkx as nx

import Embedding_models.Struc2Vec.graph as graph

logging.basicConfig(filename='struc2vec.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')


def read_graph(args_input):
    '''
    Reads the input network.
    '''
    logging.info(" - Loading graph...")
    # G = graph.load_edgelist(args_input,undirected=True)

    graph_edges = pd.read_csv(args_input, index_col = None).values.tolist()

    Graph_networkx = nx.Graph()
    list_unique_elements = {}
    location_unique_elements = {}
    i = 0 
    for elem in graph_edges:
        if not(elem[0] in list_unique_elements.keys()):
            list_unique_elements[elem[0]] = i
            location_unique_elements[str(i)] = elem[0]
            i += 1
        if not(elem[2] in list_unique_elements.keys()):
            list_unique_elements[elem[2]] = i
            location_unique_elements[str(i)] = elem[2]
            i += 1

        Graph_networkx.add_edge(list_unique_elements[elem[0]],list_unique_elements[elem[2]],label=elem[1])

    
    G = graph.from_networkx(Graph_networkx, undirected=True)

    logging.info(" - Graph loaded.")
    return G, location_unique_elements

def learn_embeddings(args, location_unique_elements):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    logging.info("Initializing creation of the representations...")
    walks = LineSentence(args.walkloc) # random_walks_wn18, random_walks_fb15k
    correct_walks = []
    for walk in walks:
        correct_walk = []
        for elem in walk:
            try:
                correct_walk.append(str(location_unique_elements[elem]))
            except:
                pass
        correct_walks.append(correct_walk)
    model = Word2Vec(correct_walks, size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.output+"struc2vec"+ str(args.dimensions) +".emb")
    logging.info("Representations created.")
    
    return

def exec_struc2vec(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    if(args.OPT3):
        until_layer = args.until_layer
    else:
        until_layer = None

    G, location_unique_elements = read_graph(args.input)
    G = struc2vec.Graph(G, args.directed, args.workers, untilLayer = until_layer)

    # if(args.OPT1):
    #     G.preprocess_neighbors_with_bfs_compact()
    # else:
    #     G.preprocess_neighbors_with_bfs()

    # if(args.OPT2):
    #     G.create_vectors()
    #     G.calc_distances(compactDegree = args.OPT1)
    # else:
    #     G.calc_distances_all_vertices(compactDegree = args.OPT1)


    # G.create_distances_network()
    # G.preprocess_parameters_random_walk()

    # G.simulate_walks(args.num_walks, args.walk_length)


    return G, location_unique_elements

def main(args):
    print("struc2vec started")
    G, location_unique_elements = exec_struc2vec(args)
    learn_embeddings(args, location_unique_elements)

