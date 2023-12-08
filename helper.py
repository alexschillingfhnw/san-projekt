import os
import sys
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite


def print_network_information(G):
    print(f"Information for given Graph with name '{G.name}':")
    print(f"\tGraph is directed: {G.is_directed()}")
    print(f"\tNumber of nodes: {G.number_of_nodes()}")
    print(f"\tNumber of edges: {G.number_of_edges()}")