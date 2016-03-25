import SimpleStrategy as ss1
import SimpleStrategy2 as ss2

class StrategyFactory(object):
    def chooseStrategy(self, strategyName,signal, price):
        if strategyName =='Simple1':
            return ss1.SimpleStrategy(signal, price)
        elif strategyName =='Simple2':
            return ss2.SimpleStrategy2(signal, price)