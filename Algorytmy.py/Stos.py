#Stos to struktura Last in first out LiFo
class Stos:
    def __init__(self,rozmiarListy):
        self._stos = list()
        self._max_elements = rozmiarListy

    def zeruj(self):
        self._stos.clear()

    def pop(self):
        if len(self._stos)>=1:
            temp = self._stos.pop()
            return temp
        else: print("pusty stos")



    def push(self,object):
        print("odkładanie na stos")
        if len(self._stos)<self._max_elements:
            self._stos.append(object)
        else: print("nie ma miejsca ")

    def __str__(self):
        if self._stos is None:
            return
        print("[",end="")
        for el in self._stos:
            print(el,end=" ")
        print("]")

    def __len__(self):
        return len(self._stos)
    def scal(self,kolejka):
        Stosx2 = Stos(self._stos.__len__()+kolejka.__len__())
        for i in range(self._stos.__len__()):
            Stosx2.push(self.pop())
        for i in range(kolejka.__len__()):
            Stosx2.push(kolejka.pop())
        return Stosx2



s = Stos(20)
s.__str__()
s.pop()
s.push(3)
s.push("cos")
s.push([3,4,12,6])
# print(s.__str__())
# print(s.__len__())

s.pop()
print(s.__str__())
s.zeruj()
s2 = Stos(10)
formula = "EASY"
word = ''
for el in formula:
    if el =="*":
        word+=s.pop()

    else:
        s.push(el)
print(s.__str__())
for i in range(s.__len__()):
    s2.push(s.pop())
s.push(3)
s.push("cos")
s.push([3,4,12,6])

print(s.__str__())
print(s2.__str__())

stosx2 = s.scal(s2)
print(stosx2.__str__())


# Koljeka Fifo first in first out

class fifo:
    def __init__(self, rozmiarListy):
        self._stos = list()
        self._max_elements = rozmiarListy
        self.licznik_ile_jest=0

    def zeruj(self):
        self._stos.clear()
    def push(self,element):
        if self._max_elements<self.licznik_ile_jest:
            self._stos.append(element)
            self.licznik_ile_jest+=1
        else:
            print("jest pełno")
    def pop(self):
        if self.licznik_ile_jest>=1:
            temp =self._stos.pop(0)
            return temp

        self.licznik_ile_jest-=1

    def __len__(self):
        return self.licznik_ile_jest
