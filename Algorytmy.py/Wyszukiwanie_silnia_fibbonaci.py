def szukaj(tabela,szukana):
    dl = len(tabela)
    for  i in range(dl):
        if  tabela[i]==szukana:
            return tabela.index(tabela[i]),len(tabela)

    return "nie ma szukanje liczby"
print("szukana wartość znajduje sie na miejscu: ",szukaj([2,3,4,5,6,7,8,0,8,12,13],12))

# szukanie dzieleniem tablicy i przeszukiwaniem binarnym
def szukanie(tab,right,left,szukana):
    if right>left:
        return "nie znaleziono"
    if tab[right]==szukana:
        return "znaleziono na miejscu ",tab.index(szukana)

    else:
        return szukanie(tab,right+1,left,szukana)
def szukanie_dzielenie(tablica,left,right,mid,szukana):
    try:
        if szukana  == tablica[int(mid)]:
            return szukana
        elif szukana<tablica[int(mid)]:
            left = 0
            right = mid
            mid  = (left+right)/2
            return szukanie_dzielenie(tablica,left,right,mid,szukana)
        elif szukana > tablica[int(mid)]:
            left = mid
            right = right
            mid = (left+right)/2
            return (szukanie_dzielenie(tablica,left,right,mid,szukana))
    except:
        return "nie znaleziono wartości"

tablica = [1,2,3,4,5,6,7,8,9,11]
left = 0
right = len(tablica)-1
mid  = int((right+left) /2)
print(f" wartość znaleziona na : {szukanie_dzielenie(tablica,left,right,mid,11)}")

tab = [2,3,4,5,6,7,8,0,8,12,13]
n = len(tab)
right = 0
print(szukanie(tab,right,n-1,6))

#oba wyniki fibbonaci zwracają to samo
def fibonnaci(a,n):
    a,b =1,1
    n=0
    while True:
        yield a,n
        a,b = b,a+b
        n+=1

for _,n in fibonnaci(1,30):
    if n ==30:
        print(_)
        break
def fibb(n):
    if n<=1:
        return 0
    list = [1,1]
    if n<=2:
        return list[:n]
    for i in range(n-1):
        list.append(list[-1]+list[-2])

    return list

print(fibb(30))

### Silnia
def silnia(n):
    a = 1
    n = 1
    while True:
        yield a,n
        a= a*n
        n+=1
for silnia,n in silnia(10):
    if n==10:
        print(silnia)
        break
silnia = 1
n =10
for i in range(2,n):
    silnia*=i

#####        dekorato
print(f"{silnia:.^20}")



#Program zwraca dla mniejszych od 100 +11  dla większych od 100 zwraca +10
def McCarthy(n):
    if n>100:
        return n-10
    else:
        return McCarthy(n+11)
print( "dla <100 +12 dla >100 -11",McCarthy(101))



