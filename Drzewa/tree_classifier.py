import pandas as pd
import numpy as np
import math
import pprint


#Drzewa są algorytmami nadzorowanymi
# drzowo tworymy poprzez podział  entropi lub wskaznika giniego
# wybieramy najlepszą opcję czyli największą entropię lub giniego
# wskaźnik ten pokazuje jak dobrze dane zostaną poidzielone w dany sposób natomiast w dalszych krokach tworzymy liście
# czyli kolejne podziały danych ze względu na inne cechy

## entropia   i stopień zaniczyszczenia   im mnijeszy tym lepiej dla klasy z podziałęm rónym jest 1/2 a my chcemy żeby było njak najmniej
#
#
#                 /                        \
#     e  =   -   | p (i,k) * log   p(ik)    |
#                 \         2              /
#

### wskaznik giniego im mnijeszy tym lepiej
#wyliczany w każdym węźle wskazuje na to ile
#
#   gini = miara podziału w danym więźle danych czyli to jak zruznicowanie dane są podzielone
# wsp giniego :
#                             /
#   gini(t)  =    1   -   sum|  [p(i,k)]^2   np dla zbioru 120 elementów podzielonego na 3 klasy po 20,50,50 elementów
#                             \
#      g  =   1 - [ (20/120)^2 + (50/120)*2  + (50/120)^2]

data = pd.DataFrame({
    "ilość": ["dużo", "średnio", "dużo", "mało", "dużo", "średnio", "mało", "dużo", "średnio", "mało",
              "mało", "dużo", "średnio", "dużo"],
    "wartość": [14, 22, 10, 30, 12, 22, 60, 40, 14, 60, 30, 25, 50, 50],
    "karany": ["TAK", "NIE", "NIE", "TAK", "TAK", "NIE", "NIE", "TAK", "NIE", "TAK",
               "NIE", "TAK", "TAK", "NIE"],
    "klasa": [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]  # Cel klasyfikacji
})
# wskaźnik giniego informuje nas jak dobrze dane zostały podzielone przy podziale czyli z jakim podobieństwem wylosujemy osoby z różnych grup
# dla 2 grup  o decyzji tak i nie  sprawdzamy jak dane będą podzielone ile tych zgrupy tak jest opisane przez atrybut jako faktycznie tak a ile pasuje pod klasę tak ale decyzja jest nie
# entropia jest miarą zanieczyszczenia danych czyli wskazuje jak zrużnicowany jest atrybut opisujący klasę dycyzji
# zysk informcyjny = entropia przed podziałem - entropia po podziale

def entropy_for_universe(data):
    label = data.keys()[-1]
    entropia_dla_zbioru = 0
    values = data[label].unique()
    for value in values:
        ilosciowo_etykieta = data[label].value_counts()[value]
        ilosciowo_uniwersum = len(data[label])
        entropia_dla_zbioru += - ilosciowo_etykieta/ilosciowo_uniwersum * math.log2(ilosciowo_etykieta/ilosciowo_uniwersum)
    return entropia_dla_zbioru



def entropy_for_attribute(data,attribute=None):
    label = data.keys()[-1]
    zero_leak =np.finfo(float).eps
    #attribute = "karany"
    target_variables = data[label].unique()
    variables = data[attribute].unique()
    entropia_klasy_decyzyjnej = 0
    for variable in variables:

        entropy_for_each_etykieta_po_podziale = 0
        for target_variable in target_variables:
            licznik = len(data[data[attribute]==variable][data[label] ==target_variable])
            mianownik = len(data[data[attribute]==variable])
            wynik = licznik/(mianownik+zero_leak)
            #print(licznik," klasa: ",target_variable," zmienna: ",variable)
            entropy_for_each_etykieta_po_podziale+= -wynik * math.log2(wynik+zero_leak)
        licznik_k_decyzji = len(data[attribute][data[attribute]==variable])
        mianownik_w_uniwersum = len(data)
        ilosc_w_calym_uniwersum = licznik_k_decyzji / mianownik_w_uniwersum
        return abs(-ilosc_w_calym_uniwersum * entropy_for_each_etykieta_po_podziale)


def find_best_split(data):
    #zbieramy wszystkie etrybuty i dla każego atrybutu dostępnego liczymy wiarygodność dla danego elementu
    Entropy_atributes =[]
    entropia_dla_zbioru = entropy_for_universe(data)
    for key in data.keys()[:-1]:
        Entropy_atributes.append((entropia_dla_zbioru - entropy_for_attribute(data,key),key))
    return max(Entropy_atributes)


def get_subtable(data,node,value):
    #podtabela czyli zbieramy wszystkie wartości dla node czyli elementu o najlepszym podziale w danej iteracji
    return data[data[node]==value].reset_index(drop=True)

def build_tree(data,tree= None):
    # zm decyzyjna
    label = data.keys()[-1]
    # pobieramy do node atrybut z najlepszym zyskiem i wartośc zysku informacji
    node = find_best_split(data)
    # dla atrybutu po którym dzielimy  pobieramy wszystkie jego wartości
    attr_value = data[node[-1]].unique()

    tree = {}
    tree[node[-1]] = {}
    print(tree)
    for value in attr_value:
        subtable = get_subtable(data, node[-1], value)
        # wartości występujące dla atrybutu, count ile jest tych przypadków
        class_value, counts = np.unique(subtable[label], return_counts=True)

        if len(counts) == 1:
            tree[node[-1]][value] = (class_value[0],f'ilosc:{counts[-1]}')
        # na zbiore
        else:
            tree[node[-1]][value] = build_tree((subtable))
    return tree
def predict(sample,tree):
    # bierzemy klucz pierwszy w drzewie które mamy
    for node in tree.keys():
        #pobieramy wartość z value
        value = sample[node]
        tree = tree[node][value]
        prediction = 0
        # jeśli obiet nie jest wartośćią tylko drzewm to wywołujemy rekurencje aż dojdziemy do value czyli decyzji dla podanych wartości
        if type(tree) is dict:
            prediction = predict(sample,tree)
        else:
            #
            prediction=tree
            break
    return prediction
#
# print(find_best_split(data))
# tree_my =build_tree(data)
# pprint.pprint(tree_my)
#
# for i in range(12):
#     sample = data.iloc[i]
#
#     pred,count = predict(sample,tree_my)
#     print(pred,end="  ")

label =data.keys()[-1]
variables =data["wartość"]
target_variables = data[label].unique()
values = variables.unique()
if len(variables) >len(values):
    for value in sorted(values):
        idx = data[data["wartość"] <= value].index

        # Count values that are <= value
        count_less_equal = len(idx)

        # Count values that are > value
        count_greater = len(data[data["wartość"] > value])

        entropy_for_each_etykieta_po_podziale = 0
        print(value,end = " ")
        for target_variable in target_variables:
            licznik1 = len(data.loc[(data["wartość"] < value) & (data[label] == target_variable)])

            mianownik = len(data[data["wartość"] < value])  # Update denominator if needed

            print(-licznik1/(mianownik+np.finfo(float).eps)*np.log2((licznik1/(mianownik+np.finfo(float).eps))+np.finfo(float).eps))



