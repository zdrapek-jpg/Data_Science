#python 3.10#!
#   RANSAC  Random Sample Consensus
#   wybieramy 2 losowe pkt liczymy dla nich parametry (x1b1+x2b2+...xnbn)+bo czyli funkcja liniowa
#   wyraz wolny robimy z podstawienie  wyliczamy przykłądową prostą i sprawdzamy jaki błąd wychodzi
# z pośrób losowań wybieramy ten któy zapewna najmnijeszy bład czyli najlepsze dopasowanie
import sys

from numpy import random, sum, square, dot, logical_not

from Data_Preprocessing.draw_2d_arrays_with_labels import draw_points_with_line


class Ransac:
    __slots__ = ["data","max_iter","distance", "min_samples", "best_model", "best_inliner_count", "inliner_mask", "show_partial_result"]
    def __init__(self,data,max_iter=None,distance=1,min_samples=0.5):
        self.data= data,
        self.max_iter = max_iter if max_iter is not None else data.shape[0]//3
        self.distance = distance   # warunek przynależności jako inliner gdy  liczyby błąd y_pred do y_actual
        self.min_samples = min_samples if 0>min_samples>0.9 and min_samples<data.shape[0] else round(min_samples*data.shape[0])   # przynajmniej połowa pkt jest jako inliner
        self.best_model = None
        self.best_inliner_count = 0
        self.inliner_mask = None

    """
    :key data = numpy array with x values and lest column y  (data[:,-1]) 
    """

    def fit(self,data=None,show_partial_result=False):
        if data is None:
            data = self.data

        x = data[:,:-1]
        y = data[:,-1]

        self.best_model = None
        self.best_inliner_count = 0
        self.inliner_mask = None
        sample_size = 2

        for i in range(self.max_iter):
            rand_idx = random.choice(data.shape[0], size=sample_size, replace=False)
            points = data[rand_idx,:]
            na_y = (points[0,-1] - points[1,-1])
            na_x = (points[0,:-1]- points[1,:-1])+sys.float_info.epsilon
            a = na_y/na_x
            b = points[0,-1]- dot(a,points[0,:-1])
            suma_wazona= (a * data[:,:-1])
            y_pred = sum(suma_wazona,axis=1)+b
            this_inlier_mask = square(y_pred-y)<self.distance
            this_inlier_count = sum(this_inlier_mask)
            better_found = (  (this_inlier_count > self.min_samples)
                              and
                              (this_inlier_count>self.best_inliner_count))
            if better_found:
                self.best_model = (a,b)
                self.best_inliner_count = this_inlier_count
                self.inliner_mask = this_inlier_mask

        if show_partial_result:
            from Linear_reg_class import LinearRegression as Ln
            my_lin = Ln(x,y)
            my_lin.fit()
            a_,b_ = my_lin.wsp_a,my_lin.wsp_b
            model = (a_,b_)
            #print(a_,b_)

            from sklearn.linear_model import RANSACRegressor
            from sklearn.linear_model import LinearRegression

            regression = RANSACRegressor(min_samples=0.3,
                                         estimator=LinearRegression(),
                                         max_trials=50,
                                         residual_threshold=350,
                                         )
            regression.fit(x, y)
            print(regression.get_params())
            inlier_mask = regression.inlier_mask_
            outlier_mask = logical_not(regression.inlier_mask_)
            a1 =regression.estimator_.coef_  # wagi dla parametrów
            b1 =regression.estimator_.intercept_  # czynnik regularyzacji
            model2 = (a1,b1)
            print(regression.inlier_mask_)

            draw_points_with_line(data,self.inliner_mask,self.best_model,model,model2)

        pass
    def predict(self,x):
        return dot(x* self.best_model[0])+self.best_model[-1]
        pass
    def coef_(self):
        return self.best_model









