import networkx as nx
from networkx.algorithms import bipartite
import random
import numpy as np
import sys
import featuregenerator


class CDConstructor:
    def __init__(self, dataset, sample_size, task = "missing_link_prediction", future_dataset = None, k_fold = 10):
        self.dataset = dataset
        self.future_dataset = future_dataset
        self.sample_size = sample_size
        self.train_sample_size = 0
        self.test_sample_size = 0
        #self.positive_examples_size = sample_proportions[0]
        #self.negative_examples_size = sample_proportions[1]
        self.edges = set()
        self.positive_examples = list()
        self.negative_examples = list()
        self.all_data = list()
        self.train_data = list()
        self.test_data = list()
        self.original_graph = None
        self.train_graph = None
        self.attributes_list = {}
        self.train_attributes_list = {}
        self.ordered_attributes_list = []
        self.train_ordered_attributes_list = []
        self.sample_dataset = None
        self.classification_dataset = None
        self.train_classification_dataset = None
        self.test_classification_dataset = None
        self.folds_indexes = {}
        self.fromnodes=[]
        self.tonodes=[]

        print('Reading data from file...')
        self.fromnodes, self.tonodes = self.read_file(self.dataset)
        self.original_graph = nx.DiGraph()
        print('Building graph...')
        for i in range(len(self.fromnodes)):
            self.original_graph.add_edge(self.fromnodes[i],self.tonodes[i])
        print('Setting negative examples...')
        self.set_negative_examples()
        print('Creating train and test data...')
        self.set_train_and_test_data()
        print('Extrating attributes of train data...')
        self.train_classification_dataset = self.calculate_train_classification_dataset()
        print('Normalizing training attributes...')
        self.normalize_attributes(self.train_classification_dataset)
        print('Extracting attributes of all data...')
        self.test_classification_dataset = self.calculate_test_classification_dataset()
        print('Normalizing all data attributes...')
        self.normalize_attributes(self.test_classification_dataset)
        print('Writing train data to train_data.csv...')
        self.write_to_csv_file(self.train_classification_dataset, 'train_data.csv', 0)
        print('Writing test data to test_data.csv...')
        self.write_to_csv_file(self.test_classification_dataset, 'test_data.csv', 0)
        print('Writing test data0 to test_data.csv...')
        self.write_to_csv_file(self.test_classification_dataset, 'test_data_no_nodes.csv', 1)

    def write_to_csv_file(self, data, filename, type):
        w = open(filename, 'w')
        if type == 0:
            for i in range(len(data)):
                for j in range(len(data[0])):
                    if j < len(data[0])-1:
                        w.write(str(data[i][j]) + ',')
                    else:
                        w.write(str(data[i][j]) + '\n')
        else:
            for i in range(len(data)):
                for j in range(2,len(data[0])):
                    if j < len(data[0])-1:
                        w.write(str(data[i][j]) + ',')
                    else:
                        w.write(str(data[i][j]) + '\n')
        w.close()

    def normalize_attributes(self, data):
        #First 2 columns are nodes, last column is Class,, others will be normalized.
        for i in range(2,len(data[0])-1):
            min_of_column = data[0][i]
            max_of_column = data[0][i]
            for j in range(len(data)):
                if data[j][i] < min_of_column:
                    min_of_column = data[j][i]
                elif data[j][i] > max_of_column:
                    max_of_column = data[j][i]
            for j in range(len(data)):
                data[j][i] = (data[j][i] - min_of_column)/(max_of_column - min_of_column)

    def set_train_and_test_data(self):
        random.shuffle(self.positive_examples)
        train_n = int(len(self.positive_examples)*4/5)
        test_n = 508837-train_n
        self.train_sample_size=train_n
        self.test_sample_size=test_n
        self.train_graph = nx.DiGraph()

        for i in self.original_graph.nodes(data=True):
            self.train_graph.add_node(i[0])

        for i in range(train_n):
            self.train_graph.add_edge(self.positive_examples[i][0],self.positive_examples[i][1])

        for i in range(2*train_n):
            self.train_data.append([])

        for i in range(2*train_n):
            if i%2 == 0:
                self.train_data[i].append(self.positive_examples[int(i/2)][0])
                self.train_data[i].append(self.positive_examples[int(i/2)][1])
            else:
                self.train_data[i].append(self.negative_examples[int(i/2)][0])
                self.train_data[i].append(self.negative_examples[int(i/2)][1])

        for i in range(2*test_n):
            self.test_data.append([])

        for i in range(2 * test_n):
            if i%2 == 0:
                self.test_data[i].append(self.positive_examples[train_n + int(i/2)][0])
                self.test_data[i].append(self.positive_examples[train_n + int(i/2)][1])
            else:
                self.test_data[i].append(self.negative_examples[train_n + int(i/2)][0])
                self.test_data[i].append(self.negative_examples[train_n + int(i/2)][1])

    def set_negative_examples(self):
        for i in range(508837):
            self.negative_examples.append([])
        for i in range(508837):
            done = False
            while done ==False:
                a = random.randint(0,75887)
                b = random.randint(0,75887)
                if (not a==b) and self.original_graph.has_node(str(a)) and self.original_graph.has_node(str(b)):
                    if not self.original_graph.has_edge(a,b):
                        self.negative_examples[i].append(a)
                        self.negative_examples[i].append(b)
                        done = True

    def calculate_test_classification_dataset(self):
        attributes_calculator = featuregenerator.FeatureConstructor(self.original_graph)
        if self.attributes_list == {}:
            self.ordered_attributes_list = sorted(attributes_calculator.attributes_map.keys())
            for attribute in self.ordered_attributes_list:
                self.attributes_list[attribute] = {}
        test_classification_dataset = np.zeros((2 * (self.train_sample_size + self.test_sample_size), len(self.train_attributes_list) + 3))
        line_count = 0
        for i in range(2 * (self.train_sample_size + self.test_sample_size)):
            if i < 2 * self.train_sample_size:
                first_node, second_node= self.train_data[i][0],self.train_data[i][1]
                attributes_calculator.set_nodes(str(first_node), str(second_node))
                column = 0
                for function in self.ordered_attributes_list:
                    parameters = self.attributes_list[function]
                    test_classification_dataset[line_count][column+2] = attributes_calculator.attributes_map[function](**parameters)
                    column += 1
                test_classification_dataset[line_count][0] = self.train_data[i][0]
                test_classification_dataset[line_count][1] = self.train_data[i][1]
                if i%2 == 0:
                    test_classification_dataset[line_count][len(self.attributes_list) + 2] = 1
                else:
                    test_classification_dataset[line_count][len(self.attributes_list) + 2] = 0
            else:
                first_node, second_node= self.test_data[i - 2*self.train_sample_size][0],self.test_data[i - 2*self.train_sample_size][1]
                attributes_calculator.set_nodes(str(first_node), str(second_node))
                column = 0
                for function in self.ordered_attributes_list:
                    parameters = self.attributes_list[function]
                    test_classification_dataset[line_count][column+2] = attributes_calculator.attributes_map[function](**parameters)
                    column += 1
                test_classification_dataset[line_count][0] = self.test_data[i - 2*self.train_sample_size][0]
                test_classification_dataset[line_count][1] = self.test_data[i - 2*self.train_sample_size][1]
                if i%2 == 0:
                    test_classification_dataset[line_count][len(self.attributes_list) + 2] = 1
                else:
                    test_classification_dataset[line_count][len(self.attributes_list) + 2] = 0
            line_count += 1
            if i%1000 ==0:
                print("Progress {:2.1%}".format(i / (2 * (self.train_sample_size + self.test_sample_size))), end="\r")

        return test_classification_dataset

    def calculate_train_classification_dataset(self):
        attributes_calculator = featuregenerator.FeatureConstructor(self.train_graph)
        if self.train_attributes_list == {}:
            self.train_ordered_attributes_list = sorted(attributes_calculator.attributes_map.keys())
            for attribute in self.train_ordered_attributes_list:
                self.train_attributes_list[attribute] = {}
        train_classification_dataset = np.zeros((2 * self.train_sample_size, len(self.train_attributes_list) + 3))
        line_count = 0
        for i in range(2 * self.train_sample_size):
            first_node, second_node= self.train_data[i][0],self.train_data[i][1]
            attributes_calculator.set_nodes(str(first_node), str(second_node))
            column = 0
            for function in self.train_ordered_attributes_list:
                parameters = self.train_attributes_list[function]
                train_classification_dataset[line_count][column+2] = attributes_calculator.attributes_map[function](**parameters)
                column += 1
            train_classification_dataset[line_count][0] = self.train_data[i][0]
            train_classification_dataset[line_count][1] = self.train_data[i][1]
            if i%2 == 0:
                train_classification_dataset[line_count][len(self.train_attributes_list) + 2] = 1
            else:
                train_classification_dataset[line_count][len(self.train_attributes_list) + 2] = 0
            line_count += 1
            if i%1000 ==0:
                print("Progress {:2.1%}".format(i / (2 * (self.train_sample_size + self.test_sample_size))), end="\r")

        return train_classification_dataset

    def calculate_classification_dataset(self):
        attributes_calculator = featuregenerator.FeatureConstructor(self.original_graph)
        if self.attributes_list == {}:
            self.ordered_attributes_list = sorted(attributes_calculator.attributes_map.keys())
            for attribute in self.ordered_attributes_list:
                self.attributes_list[attribute] = {}

        classification_dataset = np.zeros((self.sample_size, len(self.attributes_list) + 2))
        line_count = 0
        for i in range(len(self.fromnodes)):
            #first_node, second_node, pair_class, pair_fold = line
            first_node, second_node= self.fromnodes[i],self.tonodes[i]
            attributes_calculator.set_nodes(first_node, second_node)
            column = 0
            for function in self.ordered_attributes_list:
                parameters = self.attributes_list[function]
                classification_dataset[line_count][column+2] = attributes_calculator.attributes_map[function](**parameters)
                column += 1
            classification_dataset[line_count][0] = self.fromnodes[i]
            classification_dataset[line_count][1] = self.tonodes[i]
            if i%1000 == 0:
                print(i)
            line_count += 1
        w3.close()

        return classification_dataset



    def set_attributes_list(self, attributes_list):
        self.attributes_list = attributes_list
        self.ordered_attributes_list = sorted(self.attributes_list.keys())

    def read_file(self,file):
        with open(file, 'r') as f:
            data = [i.split("\t") for i in f.read().split()]

        for i in range(508837):
            self.positive_examples.append([])

        i=0
        for row in data:
            if(i==0):
                self.fromnodes.append(row[0])
                i=1
            else:
                self.tonodes.append(row[0])
                i=0

        for i in range(len(self.fromnodes)):
            self.positive_examples[i].append(self.fromnodes[i])
            self.positive_examples[i].append(self.tonodes[i])

        return self.fromnodes,self.tonodes

    def common_out_neighbors(self,g, i, j):
        return set(g.successors(i)).intersection(g.successors(j))

    def common_in_neighbors(self,g, i, j):
        return set(g.predecessors(i)).intersection(g.predecessors(j))


a=CDConstructor('social.txt',508837)
