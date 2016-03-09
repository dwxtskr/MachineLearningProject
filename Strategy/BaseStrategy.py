from abc import ABCMeta, abstractmethod


class BaseStrategy(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_position(self):
        raise NotImplementedError("Should implement generate_signals()!")








