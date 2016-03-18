from BasePortfolio import BasePortfolio
import pandas as pd
import math
import numpy as np


class RBMRandomForestPortfolio(BasePortfolio):

    def __init__(self, symbol, prices, position, initial_capital=100000.0, initial_margin=5000,maint_margin=3500, contract_size = 1000, purchase_size=1.):
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
        self.length = position.size
        self.initial_capital = float(initial_capital)
        self.initial_margin = float(initial_margin)
        self.maint_margin = float(maint_margin)
        self.contract_size = contract_size
        self.purchasing_size = purchase_size
        self.portfolio=pd.DataFrame()
        
        self.portfolio['prices'] = prices
        self.portfolio['prices_change']=self.portfolio['prices'].diff().fillna(0)
        self.portfolio['position']=position*self.purchasing_size
        self.portfolio.index = pd.to_datetime(self.portfolio.index)
        
        #annualized
        self.total_return = 0
        

    def backtest_portfolio(self):
        """
        backtesting portfolio with the generated positions
        :return:
        """
        # Create portfolio DataFrame
        account = np.ones(self.length)*self.initial_capital
        cash_added = np.zeros(self.length)
        max_cash_added=1000000000
        #realized_position = np.zeros(self.length)
        cum_pos = np.zeros(self.length)
        curr_margin = 0 
        adjusted =False
        for i in range(self.length):
            if i !=0:
                account[i]=account[i-1]+cum_pos[i-1]*self.portfolio['prices_change'][i]*self.contract_size
                # if current cumulative position is zero and no more money to pay the initial, close the portfolio
                if ((cum_pos[i-1]==0) and (account[i]+max_cash_added<self.initial_margin)):
                    account[i:]=account[i]
                    break
                # if account value less than margin requirement, add money
                if account[i]+cash_added[i]<(curr_margin):
                    if curr_margin-account[i]>max_cash_added:
                        #if no more money to add, adjust position
                          cum_pos[i]=np.sign(cum_pos[i-1])*int(max(0,(account[i]+max_cash_added))/self.maint_margin)
                          adjusted =True
                    else:
                        cash_added[i:]+=curr_margin-(account[i]+cash_added[i])
            #calculate margin need if net position increase
            margin_needed = curr_margin+abs(self.portfolio['position'][i])*self.purchasing_size*self.initial_margin
            total_account = account[i]+cash_added[i]
            if ((self.portfolio['position'][i]!=0) and (np.sign(self.portfolio['position'][i])*np.sign(cum_pos[i-1])>=0)):
                if total_account <margin_needed:
                    # if no enough fund to long/short more, maintain the position 
                    if margin_needed-account[i]>max_cash_added:
                        cum_pos[i]=cum_pos[i] if adjusted else cum_pos[i-1]
                        curr_margin=abs(cum_pos[i])*self.maint_margin
                        adjusted = False
                        continue
                    cash_added[i:]+=margin_needed-total_account 
            # recalculate  current margin
            cum_pos[i]=cum_pos[i-1]+self.portfolio['position'][i]
            curr_margin=abs(cum_pos[i])*self.maint_margin
            
        self.portfolio['portfolio']=account
        self.portfolio['Added Cash']=cash_added
        self.portfolio['Realized Cum_position']=cum_pos
        self.portfolio['Realized position']=self.portfolio['Realized Cum_position'].diff().fillna(cum_pos[0])
        self.portfolio['Total account value']=account+cash_added
        self.portfolio['P&L'] = self.portfolio['portfolio'].diff().fillna(0)
        self.portfolio['Cumulative P&L'] = self.portfolio['P&L'] .cumsum()
        self.portfolio['returns'] = self.portfolio['P&L'].div(self.portfolio['Total account value'].shift().fillna(self.initial_capital))
        self.total_return = (float(self.portfolio['Cumulative P&L'] .iloc[-1])/(self.initial_capital+cash_added[-1]))*math.sqrt(float(252*24*60)/(12500*5))

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


