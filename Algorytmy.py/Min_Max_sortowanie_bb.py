
tabela = [2,4,5,6,7,8,9,6,4,3,2,1,12,34,56,67,4,23,12,123,234,4566,675,678,7890,8900,12345,1,11,14,16,17]

def wyszukajMin(tabela):
    start = tabela[0]
    zl = 0
    for i in range(1,len(tabela)):
        zl+=1
        if start > tabela[i]:
            zl+=1
            start= tabela[i]
    return  (f"min: {start} , złożoność: {zl}")


def wyszukajMax(tabela):
    start = tabela[0]
    zl = 0
    for i in range(1,len(tabela)):
        zl+=1
        if start < tabela[i]:
            zl+=1
            start= tabela[i]
    return  (f"max: {start} , złożoność: {zl}")

print(wyszukajMin(tabela))
print(wyszukajMax(tabela))

