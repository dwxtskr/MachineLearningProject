import Strategy.SimpleStrategy as rs
import Portfolio.RBMRandomForestPortfolio as rp
import Utils as ut
import models.rbm_random_forest as rbm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import csv
import math
import numpy as np

rolling_time =10

def trans_mean_std_save(p,filename):
    p=p.transpose()
    p.columns = ['Experiment '+str(i) for i in range(rolling_time)]
    p['Mean']=p.mean(axis =1)
    p['Standard Deviation']=p.std(axis =1)
    p.to_csv(os.path.join('data', 'rbm_random_forest',filename[:2],filename+'.csv'))
    
def back_testing_portfolio(symbol, capital, initial_margin=5000,maint_margin=3500,contract_size = 1000, purchase_size = 1):

    f1_score1_pd=pd.DataFrame()
    f1_score2_pd=pd.DataFrame()
    ClassReport_pd=pd.DataFrame()
    SharpeRatio_pd=pd.DataFrame()
    AnnualReturn_pd=pd.DataFrame()
    
    for i in range(rolling_time):
        #generate signal table 
        signal_pd = ut.generate_data(symbol, i)
        
        signal_test = np.array(signal_pd['y_test'])
        signal_pred = np.array(signal_pd['y_pred'])
        
        f1_score1, f1_score2 = rbm.print_f1_score(signal_test, signal_pred)
        class_report = rbm.classification_error(signal_test, signal_pred)
        
        #generate position for predicted signal, no need for test signal
        strategy_pred = rs.SimpleStrategy(signal_pd['y_pred'])
        position_pred=strategy_pred.generate_position()

        # create MarketOpenPortfolio object
        portfolio_test = rp.RBMRandomForestPortfolio(symbol, signal_pd['price'], signal_pd['y_test'], capital, initial_margin,maint_margin, contract_size , purchase_size)
        portfolio_pred = rp.RBMRandomForestPortfolio(symbol, signal_pd['price'], position_pred, capital, initial_margin,maint_margin, contract_size , purchase_size)
        
        test_port = portfolio_test.backtest_portfolio()
        pred_port = portfolio_pred.backtest_portfolio()
        
        test_port.to_csv(os.path.join( 'data', 'rbm_random_forest','Test_Portfolio_'+symbol+'_'+str(i)+'.csv'))
        pred_port.to_csv(os.path.join( 'data', 'rbm_random_forest','Pred_Portfolio_'+symbol+'_'+str(i)+'.csv'))
        
        sharpe_test = portfolio_test.calculate_sharpe_ratio()
        sharpe_pred = portfolio_pred.calculate_sharpe_ratio()
    
        sharpe_ratio = pd.DataFrame({symbol+" Pred Sharpe Ratio":[sharpe_pred],symbol+" Test Sharpe Ratio: ":[sharpe_test]})
        annual_return = pd.DataFrame({symbol+" Pred Annualized Return":[portfolio_pred.total_return], symbol+" Test Sharpe Ratio: ":[portfolio_test.total_return]})
        
        f1_score1_pd=f1_score1_pd.append(f1_score1)
        f1_score2_pd=f1_score2_pd.append(f1_score2)
        ClassReport_pd=ClassReport_pd.append(class_report)
        SharpeRatio_pd=SharpeRatio_pd.append(sharpe_ratio)
        AnnualReturn_pd=AnnualReturn_pd.append(annual_return)

        pred_port['Cumulative P&L'].plot(label='Predict')
        test_port['Cumulative P&L'].plot(label='Test')
        plt.gcf().autofmt_xdate()
        plt.ylabel("Value ($)")
        plt.xlabel("TIme")
        plt.legend()
        plt.title(symbol+'_'+str(i)+" Cumulative P&L")
        plt.savefig(os.path.join( 'data', 'rbm_random_forest','Cum_P&L_'+symbol+'_'+str(i)+'.png'))
        plt.close()
        
        
        signal_pd['price'].plot()
        plt.gcf().autofmt_xdate()
        plt.ylabel("Value ($)")
        plt.xlabel("TIme")
        plt.legend()
        plt.title(symbol+'_'+str(i)+" Prices")
        plt.savefig(os.path.join( 'data', 'rbm_random_forest','Prices_'+symbol+'_'+str(i)+'.png'))
        plt.close()
        
    trans_mean_std_save(f1_score1_pd,symbol+'_f1_score_report_1')
    trans_mean_std_save(f1_score2_pd,symbol+'_f1_score_report_2')
    trans_mean_std_save(ClassReport_pd,symbol+'_Classification_error')
    trans_mean_std_save(SharpeRatio_pd, symbol+'_Sharpe_Ratio')
    trans_mean_std_save(AnnualReturn_pd, symbol+'_Annulized_Return')
    
        # chart
        # calculate sharpe ratio
        # compare with real signals


if __name__ == '__main__':

    back_testing_portfolio('CL', 100000, 3850,3500,1000 )
    #back_testing_portfolio('NG', 100000, 2090,1900,1000 )
    #back_testing_portfolio('GC', 100000, 4675,4250,100 )
    #back_testing_portfolio('PL', 100000, 2090,1900,50 )
    #back_testing_portfolio('HG', 100000, 3135,2850,25000)
    #back_testing_portfolio('ES', 100000, 5225,4750,50)
 

