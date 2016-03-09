from BaseStrategy import BaseStrategy
import numpy as np
import datetime

class SimpleStrategy(BaseStrategy):
    def __init__(self, signal):
        """
        Constructor of the class
        :param signal:
        """
        self.signal= np.array(signal)
        self.length = signal.size
        self.__time__ = signal.index


    def generate_position(self):
        """
        Generate signal according strategy:
        :return: position
        """
        cum_pos = 0
        position = np.zeros(self.length)
        for i in range(self.length):
            if ((self.signal[i]<0) and (cum_pos<=0)):
                # only allow short sell when at least two consecutive selling signal
                if self.signal[i-1]>=0:
                    continue
            position[i]=self.signal[i]
            cum_pos+=self.signal[i]
        return position

    #def get_time_stamp(self):
        #return self.__time__

    #def get_prices(self):
        #return self.__bars__



