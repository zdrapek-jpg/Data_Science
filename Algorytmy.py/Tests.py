class HandleError(Exception):
    def __init__(self,text,area):
        super().__init__(text)
        self.area = area
    def __str__(self):
        return ""+super().__str__() +" area :"+self.area

class HandleSecurityError(HandleError):
    def __init__(self,text,area,secure):
        super().__init__(text,area)
        self.secure  = secure
    def __str__(self):
        return super().__str__()

class HandleDataError(HandleError):
    def __init__(self,text,area,data):
        super().__init__(text,area)
        self.data=data
    def __str__(self):
        return super().__str__()




try:
    raise HandleError("fail format","data")
except HandleError as e:
    print("app error",e)


try:
    raise HandleError("fail format","data")
except HandleError as e:
    print("app error",e)


try:
    raise HandleError("fail format","data")
except HandleError as e:
    print("app error",e)
data = 123

#jak zwrucić assercje która będzie zawsze
#assert None, "zawsze"

import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
#blokowanie komunikatów o pewnym znaczeniu
#logging.disable(logging.ERROR)
# usuwa wszystkie komunikaty bez wyjątku
#logging.disable()
logging.error("błąd jest błąd xd ")
logging.critical("krytyk  xd lamusy ")

logging.basicConfig(level=logging.ERROR,format='%(asctime)s - %(levelname)s - %(message)s')
#poziomy rejestrowania
#Debug ,INFO, WARNING, ERROR, CRITICAL

import random
tab = [random.random() for i in range(20)]
logging.info(tab)

