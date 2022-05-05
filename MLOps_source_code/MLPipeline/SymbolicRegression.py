


#######################
# CREATION OF THE SYmbolic regression MODEL
#######################
#
import gplearn as gpl
from gplearn.genetic import SymbolicRegressor
import numpy as np
import time
import matplotlib.pyplot as plt

class SymbolicRegression:


    def __init__(self, df_comp):
        """
        Executing the program engine
        :param df_comp:
        """
        # xexp = gpl.functions.make_function(function=self._xexp, name='xexp', arity=1)
        # function_set = ['add', 'sub', 'mul', 'div', 'sin', 'log', xexp]
        self.function_set = ['add', 'sub', 'mul', 'div']
        self.df_comp = df_comp
        self.dataprep()
        self.train()


    def _xexp(self, x ):
        """
        It is possible to create custom operations to be considered in the tree
        :param x:
        :return:
        """
        a = np.exp(x);
        a[ np.abs(a) > 1e+9 ] = 1e+9
        return a

    def dataprep(self):
        """
        Data prep from training
        :return:
        """
        test_size = 22
        df_train = self.df_comp[:-test_size]
        df_test = self.df_comp[-test_size:]

        self.X_train = df_train[["#noofchannels", "#ofphonelines"]].values
        self.y_train = df_train["Banking"].values

        self.X_test = df_test[["#noofchannels", "#ofphonelines"]].values
        self.y_test = df_test["Banking"].values


    def train(self):
        """
        TRaining the model
        :return:
        """
        model = SymbolicRegressor(population_size=1000, tournament_size=5,
                                  generations=2, stopping_criteria=0.1,
                                  function_set=self.function_set, metric='rmse',
                                  p_crossover=0.65, p_subtree_mutation=0.15,
                                  p_hoist_mutation=0.05, p_point_mutation=0.1,
                                  verbose=1, random_state=None, n_jobs=-1)

        ###########################################################
        # TRAIN THE NETWORK AND PREDICT - Without previous values y
        ###########################################################

        # Train
        t0 = time.time()
        model.fit(self.X_train, self.y_train)
        tF = time.time()

        # Predict
        sr_y_fit = model.predict(self.X_train).reshape(-1, 1)
        sr_y_pred = model.predict(self.X_test).reshape(-1, 1)

        # Calculating Errors
        sr_residuals = sr_y_pred - self.y_test
        sr_rmse = np.sqrt(np.sum(np.power(sr_residuals, 2)) / len(sr_residuals))
        print('RMSE = %f' % sr_rmse)
        print('Time to train %.2f' % (tF - t0))
        print(model._program)

        # plot
        plt.figure(figsize=(20, 5))
        plt.plot(sr_y_pred)
        plt.plot(self.y_test, color='red')
        plt.savefig("./Output/" + "Symbolic_regression_test.png")


