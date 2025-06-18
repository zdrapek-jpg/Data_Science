import numpy as np
class LinearRegression:
    __slots__ = ["x","y","x_mean","y_mean","wsp_b","wsp_a"]
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.x_mean = []
        self.y_mean = 0

        self.wsp_b = 0
        self.wsp_a = []


    def data_mean(self, data):
        ### jeśli x shape jest pojednyńczy czyli (n,) to liczymy raz
        # jak jest (n,m) wyliczamy dla każdej kolumny wartość ścrednią
        if data.ndim==1:
            return data.mean()
            #krótszy
        # średnie po każdej kolumnie
        return data.mean(axis=0)


    def fit(self):
        self.wsp_a=0
        self.wsp_b = []
        self.x_mean = self.data_mean(self.x)
        self.y_mean  = self.data_mean(self.y)
        suma_wsp = np.zeros((self.x.shape[-1]))
        x_kwadraty = np.zeros((self.x.shape[-1]))
        # suma wyliczona dla   sxy
        for wsp_x,wsp_y in zip(self.x,self.y):
            x_srednie =wsp_x-self.x_mean
            x_kwadraty+=x_srednie**2
            y_srednie = wsp_y-self.y_mean
            y_srednie = np.full_like(x_srednie,fill_value=y_srednie)
            suma_wsp += x_srednie * y_srednie

        #print("współćzynniki przy dla x wagi to :")
        self.wsp_a= suma_wsp/x_kwadraty
        self.wsp_b = self.y_mean- sum(self.wsp_a*self.x_mean)
        return self.wsp_a,self.wsp_b


    def predict(self,x_point):
        return (x_point*self.wsp_b).sum() + self.wsp_a
    def coef_(self):
        return self.wsp_a,self.wsp_b

