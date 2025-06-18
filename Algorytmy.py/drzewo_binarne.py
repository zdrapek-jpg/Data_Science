class Wezel:
    def __init__(self,information,lewy =None, prawy=None,przodek = None):
        self.info = information
        self.lewy = None
        self.prawy=None
#  Struktora drzewa zawsze musi być uporządkowana,
#  po lewej stronie zawsze są elementy mniejsze niż po prawej
#
#

class BST_tree:
    def __init__(self):
        self.korzen =None

    def wstaw_element_do_drzewa(self,x):
        wierzcholek =Wezel(x)
        if self.korzen is None:
            self.korzen = wierzcholek
        else:
            temp = self.korzen
        while True:
            rodzic = temp
            if x <temp.klucz:
                temp = temp.lewy
                if temp ==None:
                    rodzic.lewy = wierzcholek
                    break
            else:
                temp=temp.prawy
                if temp==None:
                    rodzic.prawy =x
                    break

    def szukaj_elementu(self,x):
        if self.korzen ==None:
            return None
        temp = self.korzen
        while temp.klucz!=x:
            if temp.klucz>x:
                temp = temp.lewy
            elif temp.klucz<x:
                temp= temp.prawy
            elif temp ==None:
                return None
        return temp
#                korzen      = temp.klucz
#                  klucz
#                /     \
#            lewy       prawy     temp.lewy  / temp.prawy
#            klucz       klucz
#           /    \       /   \
#         lewy   prawy  lewy  prawy
#        klucz    klucz  klucz klucz
#        /  \
#     klucz

