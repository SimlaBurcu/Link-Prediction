import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import inspect
import random

class FeatureConstructor:
    def __init__(self, graph, node1=None, node2=None):
        self.graph = graph
        self.node1 = node1
        self.node2 = node2
        self.average_neighbor_degree_dict = None

        self.attributes_map = {
            "adamic_adar_similarity": self.adamic_adar_similarity,
            "average_neighbor_degree_sum": self.average_neighbor_degree_sum,
            "common_neighbors": self.common_neighbors,
            "cosine": self.cosine,
            "jaccard_coefficient": self.jaccard_coefficient,
            "preferential_attachment": self.preferential_attachment,
            "sum_of_neighbors": self.sum_of_neighbors,
        }

        if(self.node1 != None and self.node2 != None):
            self.neighbors1 = self.all_neighbors(self.node1)
            self.neighbors2 = self.all_neighbors(self.node2)

    def set_nodes(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.neighbors1 = self.all_neighbors(self.node1)
        self.neighbors2 = self.all_neighbors(self.node2)
        
    def all_neighbors(self, node):
        neighbors = set()
        print(nx.all_neighbors(self.graph, node))
        for neighbor in list(nx.all_neighbors(self.graph, node)):
            neighbors.add(neighbor)
        return neighbors - set([node])

    def common_neighbors(self):
        return len(self.neighbors1.intersection(self.neighbors2))

    def sum_of_neighbors(self):
        return len(self.neighbors1) + len(self.neighbors2)

    def jaccard_coefficient (self):
        try:
            return len(self.neighbors1.intersection(self.neighbors2))/len(self.neighbors1.union(self.neighbors2))
        except ZeroDivisionError:
            return 0

    def adamic_adar_similarity(self):
        measure = 0
        for neighbor in self.neighbors1.intersection(self.neighbors2):
            secondary_neighbors = self.all_neighbors(neighbor)
            measure += 1 / (np.log10(len(secondary_neighbors)))
        return measure

    def preferential_attachment(self):
        return len(self.neighbors1) * len(self.neighbors2)

    def cosine(self):
        try:
            return self.common_neighbors() / self.preferential_attachment()
        except ZeroDivisionError:
            return 0

    def average_neighbor_degree_sum(self):
        if (self.average_neighbor_degree_dict == None):
            self.average_neighbor_degree_dict = nx.average_neighbor_degree(self.graph)
        return self.average_neighbor_degree_dict[self.node1] + self.average_neighbor_degree_dict[self.node2]
