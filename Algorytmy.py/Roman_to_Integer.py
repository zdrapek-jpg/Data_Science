from typing import Any

roman_numerals = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000
}

s = "VII"
sum = 0
i=0
length = len(s)
while i <length:
    first= roman_numerals.get(s[i])
    if  i<length-1  and roman_numerals.get(s[i+1])>first :
        sum+=roman_numerals.get(s[i+1])-first
        i+=2
    else:
        sum+=(first)
        i+=1
    print(sum)
#print(sum)



def count_discs(discs: tuple[int, ...]) -> int:
    length = len(discs)-2
    counter=1
    max = discs[-1]
    first_one = discs[-1]
    while length>=0:
        if discs[length]>max:
            max = discs[length]
            counter+=1
        length-=1

    return counter

print("Example:")
# print(count_discs((3, 2)))
#
# # These "asserts" are used for self-checking
# print(count_discs((3, 6, 7, 4, 5, 1, 2)))
# print(count_discs((6, 5, 4, 3, 2, 1)))
# print(count_discs((5,)))
# od 5 do 9 jest 1 od 10 do 15 jest 2 i od 16 do 20 jest zwiększamy co 5 ilość zer
n = 38
def zeros_in_n(n):
    zeros =0

    zeros += n//5

    for i in range(0,n//2):
        if 5**i<=n:
            zeros+=n//(5**i)
        else:
            break
    return zeros
print(zeros_in_n(100))


def flat_list(lista):
    big= []
    if not isinstance(lista,list):
        big.extand(lista)
    else:
        for el in lista:
            if isinstance(el,list):
                big.extand(flat_list(el))
            else:
                big.append(el)
    return big



#ghost opacity

def checkio(opacity):
    start = 10000
    breakp = start-opacity


    fib= [1,1]
    i = 2
    while fib[-1]<breakp:
        fib.append(fib[i - 1] + fib[i - 2])
        i+=1
    fib = set(fib)
    years = 0
    if breakp==1:
        return 1
    i=1
    while not (opacity== start):
        if opacity== start:
            break
        if i in fib:
            start-= i
        else:
            start+=1
        years += 1
        i+=1
    if years>=5000:
        return  None
    if opacity>start:
        return years-1
    return years



    #while start>opacity:
    print(fib)


#
# print(checkio(10000))
# print(checkio(9999))
# print(checkio(9997))
# print(checkio(9994))
# print(checkio(9995))
# print(checkio(9990))
# print(checkio(6736))



def min(*args, **kwargs):
    key = kwargs.get("key", None)
    minimum = args[0]
    for el in args:
        if minimum >el :
            minimum =el
    return minimum



def max(*args, **kwargs):
    key = kwargs.get("key", None)
    maximum = args[0]
    for el in args:


        if maximum<el :
            maximum =el
    return maximum
#
# print(min(1,2,3,45))
#
# print(min("asvdkf"))
#
# print(max(1,2,3,45))
#
# print(max("sskjacz"))
def canConstruct(ransomNote, magazine):
    """
    :type ransomNote: str
    :type magazine: str
    :rtype: bool
    """
    magazine = list(magazine)

    for j in range(len(ransomNote)):
        f = False
        for el in magazine:
            if ransomNote[j] == el:
                magazine.remove(el)
                f =True
                break
        if f ==False:
            return False
    return True
#
# print(canConstruct("aaA","aaa"))
# print(canConstruct("aa","aAAaBC"))
# print(canConstruct("caaa","aaa"))


def dadagram(word,dada):
    if len(word)!= len(dada):
        return False
    dada = list(dada)


    for j in range(len(word)):
        f = False
        for el in dada:
            if word[j] == el:
                dada.remove(el)
                f =True
                break
        if f ==False:
            return False
    return True

#print(dadagram("kupa","pakuu"))
def sum_by_types(items: list[str, int]) -> tuple[str, int] | list[str, int]:
    dc = {"stringi":"","liczby":0}
    for i in items:
        if isinstance(i,str):
            dc["stringi"]+=i
        else:
            dc["liczby"]+=i
    return dc.values()
#
# print("Example:")
# print(list(sum_by_types([])))
#
# print(sum_by_types([]))
# print(sum_by_types([1, 2, 3]))
# print(sum_by_types(["1", 2, 3]))
# print(sum_by_types(["1", "2", 3]))
# print(sum_by_types(["1", "2", "3"]))
# print(sum_by_types(["size", 12, "in", 45, 0]))
#
# print("The mission is done! Click 'Check Solution' to earn rewards!")
#
#
def calculator(log: str) -> int | Any:
    log = log.split("=")
    for i in range(len(log)):
        log[i]=log[i].lstrip('0')
    print(log[-1])
    try:
        return eval(log[-1])
    except:
        return log[-1]


print("Example:")
print(calculator("1+2"))

# These "asserts" are used for self-checking
print(calculator("000000"))
print(calculator("0000123"))
print(calculator("12"))
print(calculator("+12"))
print(calculator(""))
print(calculator("1+2"))
print(calculator("2+"))
print(calculator("1+2="))
print(calculator("1+2-"))
print(calculator("1+2=2"))
print(calculator("=5=10=15"))