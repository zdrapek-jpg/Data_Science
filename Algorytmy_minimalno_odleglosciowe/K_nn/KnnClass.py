import numpy as np
from Data_Preprocessing.QuickSort import  quickSort

class Knn:
    def __init__(self,data,classified_point,p=1):
        if p!=2:
            self.p = 2
        else: self.p = p
        self.data=data
        self.classified_point=classified_point
        self.classes = dict()

    def get_num_of_categories(self,labels):
        count_categories = 0
        for el in labels:
            if el in self.classes.keys():
                continue
            if el not in self.classes.keys():
                count_categories += 1
                self.classes[el]=0
        return count_categories


    def classify_point_at_k_neightbours(self,x,y,point_for_classification,k):
        count_categories = self.get_num_of_categories(y.tolist())
        distances_to_klasses = self.sort_array_of_distances(self.metric(point_for_classification, x, y))
        sortedDistances_to_klasses = self.sort_array_of_distances(distances_to_klasses)

        distance = 0
        i= 0
        while i<=len(sortedDistances_to_klasses) and i<k:
            distance = sortedDistances_to_klasses[i][0]
            klasa = sortedDistances_to_klasses[i][1]
            if klasa in self.classes.keys():
                self.classes[klasa] +=1/distance**2
            elif klasa not in self.classes.keys():
                self.classes[klasa]= 1/distance**2

            i+=1
        # klasę przypiszemy tam gdzie jest większa suma wag
        # w metodzie kwadratów odwrotności nie liczymy średniej
        # w sumie dystansów liczymy średnią
        print(f" closest to object by {k} points  is :")
        max_val = max(self.classes.values())
        decision ={key: round(value,4) for key, value in self.classes.items() if value == max_val}
        print(decision)
        return decision


    def add_pred_point(self):
        pass
    #CITY BLOCK
    def metric(self,added_point,x,y):

        if len(added_point)!= x.shape[-1]:
            print("exc")
            #raise f"mismatch data distances with {len(added_point)} != {distances.shape[-1]}"
        sumy_dystansow = []
        for point,label in zip(x,  y):
            row =sum(abs((added_point - point)) ** self.p)**(1/self.p)
            row = np.append(row,label)
            sumy_dystansow.append(row)
        return np.array(sumy_dystansow)

    def sort_array_of_distances(self,array):
        array = quickSort(array,0,len(array)-1)
        #print(array)
        return array

