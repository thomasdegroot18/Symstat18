#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time

from joblib import Parallel, delayed
from gensim.models.word2vec import logger, FAST_VERSION
import scipy


import Embedding_models.Diff2Vec.diffusion_2_vec as diffmain 
import Embedding_models.Node2Vec.main as nodemain 
import Embedding_models.Struc2Vec.main as strucmain


########################################### EmbeddingsLearning ######################################################



def learn_embeddings(args, walks_diff, location_unique_elements):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
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
    walks = correct_walks + walks_diff
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.output+"Infuser"+ str(args.dimensions) +".emb")
    
    return


############################# MAIN #######################################################



def main(args):
    print("Infuser started")
    walks_diff = diffmain.run_parallel_feature_creation(args.input,  args.vertex_set_cardinality, args.num_diffusions, args.workers, args.type)
    walks_diff = [w for walk in walks_diff for w in walk]

    G, location_unique_elements = strucmain.exec_struc2vec(args)

    learn_embeddings(args, walks_diff, location_unique_elements)

