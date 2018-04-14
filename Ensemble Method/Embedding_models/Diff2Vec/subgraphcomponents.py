from Embedding_models.Diff2Vec.diffusiontrees import EulerianDiffusionTree, EndPointDiffusionTree
import pandas as pd
import networkx as nx
import time
import random
import numpy as np

class SubGraphComponents:
    
    def __init__(self, edge_list_path, seeding):
        
        self.seed = seeding
        self.start_time = time.time()
        graph_edges = pd.read_csv(edge_list_path, index_col = None).values.tolist()


        self.graph = nx.Graph() 
        for elem in graph_edges:
            self.graph.add_edge(elem[0],elem[2],label=elem[1])

 
    def separate_subcomponents(self):
        
        self.graph = sorted(nx.connected_component_subgraphs(self.graph), key = len, reverse = True)
        
    def print_graph_generation_statistics(self):       
        print("The graph generation at run " + str(self.seed) + " took: " + str(round(time.time() - self.start_time, 3)) + " seconds.\n") 
        
    def single_feature_generation_run(self, vertex_set_cardinality, traceback_type):
        
        random.seed(self.seed)
        
        self.start_time = time.time()
        
        self.paths = {}

        for sub_graph in self.graph:
 
            nodes = sub_graph.nodes()
            random.shuffle(list(nodes))
            
            current_cardinality = len(nodes)
            
            if current_cardinality < vertex_set_cardinality:
                vertex_set_cardinality = current_cardinality
            for node in nodes:
                tree = EulerianDiffusionTree(node)
                tree.run_diffusion_process(sub_graph, vertex_set_cardinality)
                path_description = tree.create_path_description(sub_graph)
                self.paths[node] = list(map(lambda x: str(x), path_description))
                
        self.paths = list(self.paths.values())
                
    def print_path_generation_statistics(self):
        print("The sequence generation took: " + str(time.time() - self.start_time))
        print("Average sequence length is: " + str(np.mean(list(map(lambda x: len(list(x)), self.paths)))))
        
    def get_path_descriptions(self):
        return self.paths
