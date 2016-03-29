from BaseStrategy import BaseStrategy
import numpy as np
import datetime

class SimpleStrategy(BaseStrategy):
    def __init__(self, signal,price):
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
        curr_pos = 0
        position = np.zeros(self.length)
        
        for i  in range(self.length):
            if self.signal[i]==np.sign(curr_pos)*(-1):
                position[i]=self.signal[i]-curr_pos
                curr_pos = self.signal[i]
            else:
                position[i]=self.signal[i]
                curr_pos+=self.signal[i]
        return position

    #def get_time_stamp(self):
        #return self.__time__

    #def get_prices(self):
        #return self.__bars__



