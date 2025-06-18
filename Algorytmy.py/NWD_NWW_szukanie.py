#największy wspólny dzielnik, i najmniejsza spólna wielokrotność
def nwd(a,b):
    if a==0:
        return b
    if b==0:
        return a
    else:
        return nwd(b,a%b)

def nww(a,b):
   return (a*b)/nwd(a,b)

print(nwd(10,20))
print(nww(15,125))

