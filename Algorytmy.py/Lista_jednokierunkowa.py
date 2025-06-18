import logging
#logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(message)s')
class Element_Listy():
    def __init__(self,data=None,nastepny=None):
        self.data=data
        self.nastepny = nastepny

class Lista():
    def __init__(self,glowa=None,wskazanie_na_koniec_kolejki=None):
        self.glowa = glowa
        self.wskazanie_na_koniec_kolejki = wskazanie_na_koniec_kolejki
        self.dlugosc = 0
    def insert_toList(self,dane):
        data = Element_Listy(dane)
        logging.info(f"dane : {data.data},nastepny: {data.nastepny}")
        if self.glowa==None:
            self.glowa = data
            self.wskazanie_na_koniec_kolejki = data
            #logging.info(f"dane zaczyna od głowy i jest  {self.glowa.data} kolejny {self.glowa.nastepny} odwolanie do konca kolejki :{self.wskazanie_na_koniec_kolejki.nastepny}")
        else:
            self.wskazanie_na_koniec_kolejki.nastepny=data
            self.wskazanie_na_koniec_kolejki =data
            #logging.info(f"dane elementu to (kolejny) {self.wskazanie_na_koniec_kolejki.data}, {self.wskazanie_na_koniec_kolejki.nastepny} odwolanie do następnego to :{self.wskazanie_na_koniec_kolejki.nastepny}")
        self.dlugosc+=1
    def __len__(self):
        return self.dlugosc
    def __str__(self):

        adres_stmp = self.glowa
        if adres_stmp==None:
            return None
        print("[",end="")
        while adres_stmp != None:
            print(adres_stmp.data,end=", ")
            adres_stmp = adres_stmp.nastepny

        print("]")
    def szukaj_elementu(self,x):
        adres_stmp = self.glowa
        i = 0
        while adres_stmp != None:
            logging.info(
                f"dane zaczyna od głowa {adres_stmp.data} kolejny {adres_stmp.nastepny}")
            if adres_stmp.data == x:
                print("znaleziony element na indexie:", i)

                break
            else:
                adres_stmp = adres_stmp.nastepny
            if adres_stmp == None:
                print("nie znaleziono ")
                break
            i += 1
    def length(self):
        data_stmp = self.glowa
        if data_stmp==0:
            return 0
        length=0
        while data_stmp!=None:
            length+=1
            data_stmp=data_stmp.nastepny
        return length
    def delete(self,x_data):
        if self.glowa.data is None:
            raise "metoda wywołana na pustej kolejce"

        #usuwanym elementem jest pierwszy czyli glowa
        if self.glowa.data==x_data:
            self.glowa =self.glowa.nastepny
            return

        data_stmp = self.glowa
        while data_stmp.nastepny!=None:

            if data_stmp.nastepny.data==x_data:

                # usunięcie ostatniego w kolejce
                if data_stmp.nastepny.nastepny is None:
                    data_stmp.nastepny = None
                    self.wskazanie_na_koniec_kolejki = data_stmp
                    return

                #usunięcie srodkowego elementu
                data_stmp.nastepny = data_stmp.nastepny.nastepny
                return
            #przeskakiwanie na kolejny obiekt
            data_stmp = data_stmp.nastepny

    def wstaw_z_sorowaniem(self,dane_do_wstawienia):
        data = Element_Listy(dane_do_wstawienia)
        #logging.info(f"dane : {data.data},nastepny: {data.nastepny}")
        if self.glowa is None:
            self.glowa = data
            self.wskazanie_na_koniec_kolejki = data
            return
        flaga_szukania =True
        element_przed = None
        element_po = self.glowa
        while flaga_szukania and not(element_po is  None):

            if element_po.data>=dane_do_wstawienia:
                flaga_szukania=False
            #wstawienie na początek listy przed glową czyli wskazanie na koniec nie przechodzi
            # element przed wskazuje na element którym była głowa
            else:
                element_przed= element_po
                element_po = element_po.nastepny
                #print(f"porównuje {element_przed.data} z {data.data}")


        if element_przed is None:
            self.glowa=data
            data.nastepny=element_po
        else:
            #koniec listy
            if element_po is None:
                element_przed.nastepny=data
                self.wskazanie_na_koniec_kolejki=data
            #srodek listy
            else:
                element_przed.nastepny =data
                data.nastepny =element_po

    def merge_2_lists(self,lista):
        merged_list = Lista()
        lista1 = self.glowa
        lista2 = lista.glowa
        while(lista1!=None ):
            merged_list.wstaw_z_sorowaniem(lista1.data)
            lista1=lista1.nastepny
        while (lista2 != None):
            merged_list.wstaw_z_sorowaniem(lista2.data)
            lista2 = lista2.nastepny
        return merged_list
    def __iter__(self):
        return MojIterator(self)
class MojIterator:
    def __init__(self,plista):
        self._kursor =plista.glowa

    def __next__(self):
        if self._kursor  is  not None:
            if self._kursor.nastepny is None:
                res = str(self._kursor.data)
            else:
                res = str(self._kursor.data)+"-"



            self._kursor= self._kursor.nastepny
            return res
        else:
            raise StopIteration






def sortuj(a,b):
    if a==None:
        return b
    if b==None:
        return a
    if a.data<b.data:
        a.nastepny = sortuj(a.nastepny,b)
        return a
    else:
        b.nastepny = sortuj(a,b.nastepny)
        return b
def fuzja(lista1,lista2):
    nowa_lista = Lista()
    nowa_lista.dlugosc = lista1.dlugosc+lista2.dlugosc
    nowa_lista.glowa = sortuj(lista1.glowa,lista2.glowa)
    if lista1.glowa is None:
        nowa_lista.wskazanie_na_koniec_kolejki=lista2.wskazanie_na_koniec_kolejki
    elif lista2.glowa is None:
        nowa_lista.wskazanie_na_koniec_kolejki=lista1.wskazanie_na_koniec_kolejki
    else:
        if lista1.wskazanie_na_koniec_kolejki is None:
            nowa_lista.wskazanie_na_koniec_kolejki=lista1.wskazanie_na_koniec_kolejki
        else:
            nowa_lista.wskazanie_na_koniec_kolejki = lista2.wskazanie_na_koniec_kolejki
    return nowa_lista




#glowa -> [data: 3, nastepny: [data: 6, nastepny: [data: 12, nastepny:[data=13, nastepny:None]]]]
#wskazanie_na_kolejny -> [data: 12, nastepny: None]
#dlugosc = 3
l= Lista()
l.insert_toList(3)
l.insert_toList(6)
l.insert_toList(12)
l.insert_toList(13)
#print(l.__str__())
print(l.length())
print(l.delete(3))
#print(l.__str__())
# print(l.delete(12))
# print(l.__str__())
print(l.delete(13))
#print(l.__str__())
print(l.length())
print(l.insert_toList(18))
#print(l.__str__())

l2 = Lista()
#print(l2.__len__())
l2.wstaw_z_sorowaniem(12)
l2.wstaw_z_sorowaniem(10)
l2.wstaw_z_sorowaniem(11)
l2.wstaw_z_sorowaniem(15)
#print(l2.__str__())
# łączenie tablic
# merged_list = l2.merge_2_lists(l)
# print(merged_list.__str__())
# x = 12
# fuzed_lists = fuzja(l,l2)
# print(fuzed_lists.__str__())

# iterowanie po obiektach
# iterator= iter(l)
# while True:
#     try:
#         res=next(iterator)
#         print(res,end=" ")
#     except StopIteration:
#         break

