from BasePortfolio import BasePortfolio
import Strategy.StrategyFactory as sf
import pandas as pd
import math
import numpy as np


class RBMRandomForestPortfolio(BasePortfolio):

    def __init__(self, strategyName, symbol, prices, signal, initial_capital=100000.0, initial_margin=5000,maint_margin=3500, contract_size = 1000, purchase_size=1.):
        """
        Construct of the class
        :param symbol: symbol of the contract
        :param length: length of the portfolio dataframe
        :param position:
        :param initial_capital:
        :param initial_margin: amount need to move to margin account per contract long/short
        :param initial_margin: amount need in the margin account per contract hold
        :param contract_size: number of units in one contract
        :param purchasing_size: number of contract long/short each time
        :param portfolio: 
        :param portfolio['position']: long/short position for each period
        :param portfolio['Cum_Position']: cummulative net position by the end of the period
        :param total_return: total annualized return over the period
        """
        self.symbol = symbol
        self.length = signal.size
        self.initial_capital = float(initial_capital)
        self.initial_margin = float(initial_margin)
        self.maint_margin = float(maint_margin)
        self.contract_size = contract_size
        self.purchasing_size = purchase_size
        self.portfolio=pd.DataFrame()
        
        self.portfolio['prices'] = prices
        self.portfolio['prices_change']=self.portfolio['prices'].diff().fillna(0)
        self.portfolio['position'] = sf.StrategyFactory().chooseStrategy(strategyName,signal, prices).generate_position()*self.purchasing_size
        self.portfolio['Cum_position'] = self.portfolio['position'].cumsum().shift(1).fillna(0)
        self.portfolio.index = pd.to_datetime(self.portfolio.index)
        
        #annualized
        self.total_return = 0
        

    def backtest_portfolio(self):
        """
        backtesting portfolio with the generated positions
        :return:
        """
        self.portfolio['P&L'] = self.portfolio['prices'].diff().fillna(0).multiply(self.portfolio['Cum_position'])
        self.portfolio['Cumulative P&L'] = self.portfolio['P&L'] .cumsum()
        self.portfolio['portfolio']=self.initial_capital+self.portfolio['Cumulative P&L']
        self.portfolio['Margin_Account']=(np.sign(self.portfolio['Cum_position'].multiply(self.portfolio['position']))>=0).multiply( \
                                                            np.abs(self.portfolio['position']))*self.initial_margin + self.portfolio['Cum_position']* self.maint_margin
        self.portfolio['Added Cash'] = np.maximum(self.portfolio['Margin_Account']-self.portfolio['portfolio'],0)
        self.portfolio['Total account value']=self.portfolio['portfolio'] + self.portfolio['Added Cash']
        
        self.portfolio['returns'] = self.portfolio['P&L'].div(self.portfolio['Total account value'].shift().fillna(self.initial_capital))
        self.total_return = (float(self.portfolio['Cumulative P&L'] .iloc[-1])/(self.initial_capital+self.portfolio['Added Cash'][-1]))*math.sqrt(float(252*24*60)/(12500*5))

        return self.portfolio
        
    def daily_return(self):
        #print type(self.portfolio.index[0])
        d_return=np.array([])
        curr_day = self.portfolio.index[0].day
        index_start = 0
        for i in range(self.length):
            if self.portfolio.index[i].day!=curr_day :
                d_return=np.append(d_return, float(self.portfolio['portfolio'].iloc[i-1]-self.portfolio['portfolio'].iloc[index_start])/ \
                (self.portfolio['portfolio'].iloc[index_start]+self.portfolio['Added Cash'].iloc[i-1]))
                curr_day = self.portfolio.index[i].day
                index_start = i-1
        d_return=np.append(d_return, float(self.portfolio['portfolio'].iloc[-1]-self.portfolio['portfolio'].iloc[index_start])/ \
        (self.portfolio['portfolio'].iloc[index_start]+self.portfolio['Added Cash'].iloc[-1]))
        return d_return
            
    def calculate_sharpe_ratio(self):
        """
        calculate Sharpe Ratio against input benchmark
        :param bmk:
        :return:
        """
        d_return = self.daily_return()
        sharpe_ratio = d_return.mean() / d_return.std() * math.sqrt(252)
        return sharpe_ratio


