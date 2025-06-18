import random

data = [random.randrange(1,100) for i in range(20)]

def bubble_sort(data,):

    flag = True
    for i in range(0,len(data)):
        flag =False
        for j in range(i+1,len(data)):
            if data[i]>data[j]:
                data[i],data[j] = data[j],data[i]
    print(data)


from Lista_jednokierunkowa import Lista
def Lista_jednokierunkowa(data):
    L= Lista()
    for i in data:
        L.wstaw_z_sorowaniem(i)
    print()
    return L.__str__()

def quick_sort(data):
    for i in range(0,len(data)):
        max_data = data[i]
        for j in range(i,len(data)):
            if max_data >=data[j]:
                max_data=data[j]
                idx_zamiana= j

        data[i],data[idx_zamiana] = max_data,data[i]
    print(data)
from heaping import heap
def heaping(data):
    newheap = heap(30)
    for x in data:
        newheap.wstaw(3)
        newheap.wstaw(4)
        newheap.wstaw(12)
        newheap.wstaw(1)
        newheap.wstaw(19)
        newheap.wstaw(31)
    print(newheap.licznik, newheap.rozmiar)
    print(newheap.__str__())
    newheap.wypisz()

heaping(data)
# print()
# quick_sort(data)
# Lista_jednokierunkowa(data)
# bubble_sort(data)







