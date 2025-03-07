import numpy as np
from collections import defaultdict
from math import sqrt

class AANG:
    def __init__(self,x,y,k=2):
        self.x= x
        self.y=y
        self.k=k

    def similarity(self,a,b):
        distance = sum([(x1-x2)**2 for x1,x2 in zip(a,b)])
        sim = np.exp(-sqrt(distance))
        return sim
    def build_aang_graph(self):
        graph = defaultdict(list)

        for i,point in enumerate(self.x):
            similarities = []
            for j,point_ in enumerate(self.x):
                if i!=j:
                    sim = self.similarity(point,point_)
                    similarities.append((i,sim))
            similarities.sort(key=lambda x: x[1],reverse=True)
            for j, sim in similarities[:self.k]:
                graph[i].append((j, sim))  # Store index and similarity as an edge

        return graph



    def propagation_activation(self,graph):
        predictions =[]
        for i in range(len(self.x)):
            neighbor_labels = []
            for j,_ in graph[i]:
                neighbor_labels.append(self.y[j])
            majority_label = np.argmax(np.bincount(neighbor_labels))
            predictions.append(majority_label)
        return predictions

    def classify_point(self,new_point,graph):
        similarities = []
        for i in range(len(self.x)):
            similarities.append((i,self.similarity(new_point,self.x[i])))
        similarities.sort(key =lambda x: x[1],reverse=True)
        neighbor_labels = [self.y[i] for i, _ in similarities[:self.k]]
        print(neighbor_labels)
        predicted_label = np.argmax(np.bincount(neighbor_labels))

        return predicted_label


