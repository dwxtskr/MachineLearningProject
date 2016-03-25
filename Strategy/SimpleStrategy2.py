from BaseStrategy import BaseStrategy
import numpy as np
import datetime

class SimpleStrategy2(BaseStrategy):
    def __init__(self, signal, price):
        """
        Constructor of the class
        :param signal:
        """
        self.signal= np.array(signal)
        self.length = signal.size
        self.__time__ = signal.index
        self.price = price

    def generate_position(self):
        """
        Generate signal according strategy:
        :return: position
        """
        position = np.zeros(len(self.signal))
        for i in range(len(self.signal)):
            if i!=0:
                if self.signal[i]==0:
                    self.signal[i]=self.signal[i-1]
        return np.append(self.signal[0],np.diff(self.signal))

    #def get_time_stamp(self):
        #return self.__time__

    #def get_prices(self):
        #return self.__bars__



