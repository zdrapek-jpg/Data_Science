import random


class heap:
    def __init__(self,rozmiar_sterty=1):
        self.rozmiar = rozmiar_sterty
        self.licznik = 0
        self._sterta = [0 for i in range(rozmiar_sterty+1)]
    def wstaw_element(self,x):
        self.licznik+=1
        self._sterta[self.licznik]= x
        self.po_wstawieniu_do_gory()

    def zdejmij_element(self):
        x= self._sterta[1]
        self._sterta[1]= self._sterta[self.licznik]
        self.licznik-=1
        self.po_zdjeciu_na_dol()
        return x
    def po_zdjeciu_na_dol(self):
        # zrzucając element sprawdzamy czy element jest mniejszy od lewego lub prawego i wtedy idzemy dalej
        wierzcholek = 1
        actual = self._sterta[1]
        while wierzcholek< self.licznik :
            n = 2*wierzcholek
            if n >self.licznik:
                break
            if n+1 <=self.licznik:
                if self._sterta[n]<self._sterta[n+1]:
                    n = n+1
            if self._sterta[wierzcholek]>=self._sterta[n]:
                break
            self._sterta[wierzcholek],self._sterta[n] = self._sterta[n],self._sterta[wierzcholek]
            wierzcholek = n











    def po_wstawieniu_do_gory(self):
        temp = self._sterta[self.licznik]
        n = self.licznik
        while n!=1 and  self._sterta[n//2]<self._sterta[n]:
            self._sterta[n]=self._sterta[n//2]
            n = n//2
        self._sterta[n]=temp

    def __len__(self):
        return self.licznik

    def __sizeof__(self):
        return self.rozmiar

    def __str__(self):
        if self.licznik<=0:
            print( "pusto")
        n =1
        while self.licznik//2>=n:
            print(f"rodzic: {self._sterta[n]} dziecko lewe: {self._sterta[n*2]}",end="")
            if n*2+1<self.licznik:
                print(f"dziecko prawe: {self._sterta[n*2+1]}")
            n+=1
new_heap = heap(11)
x= [2,4,5,6,7,8,1,9,10]
print(new_heap.licznik)
print(x)
for el in x:
    new_heap.wstaw_element(el)
new_heap.zdejmij_element()
for i in range(new_heap.licznik):
    print(new_heap.zdejmij_element(),end = " ")
new_heap.zdejmij_element()
new_heap.__str__()
####
#                  23
#                /    \
#              18      10
#            /   \     /  \
#          12     13   7    8
#         /  \     /
#        11   2    5

