import Strategy.SimpleStrategy2 as rs
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

def trans_mean_std(p, symbol):
    p=p.transpose()
    p.columns = ['Experiment '+str(i) for i in range(rolling_time)]
    mulindex = pd.MultiIndex.from_product([[symbol],list(p.index)], names = ['Symbol', 'Type'])
    p.index = mulindex 
    p['Mean']=p.mean(axis =1)
    p['Standard Deviation']=p.std(axis =1)
    return p
    #p.to_csv(os.path.join('data', 'rbm_random_forest',filename[:2],filename+'.csv'))
    
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
        

        strategy_pred = rs.SimpleStrategy2(signal_pd['y_pred'])
        strategy_test = rs.SimpleStrategy2(signal_pd['y_test'])
        
        
        position_pred=strategy_pred.generate_position()
        position_test=strategy_test.generate_position()

        # create MarketOpenPortfolio object
        portfolio_test = rp.RBMRandomForestPortfolio(symbol, signal_pd['price'], position_test, capital, initial_margin,maint_margin, contract_size , purchase_size)
        portfolio_pred = rp.RBMRandomForestPortfolio(symbol, signal_pd['price'], position_pred, capital, initial_margin,maint_margin, contract_size , purchase_size)
        
        test_port = portfolio_test.backtest_portfolio()
        pred_port = portfolio_pred.backtest_portfolio()
        
        test_port.to_csv(os.path.join( 'data', 'rbm_random_forest',symbol,'Test_Portfolio_'+symbol+'_'+str(i)+'.csv'))
        pred_port.to_csv(os.path.join( 'data', 'rbm_random_forest',symbol,'Pred_Portfolio_'+symbol+'_'+str(i)+'.csv'))
        
        sharpe_test = portfolio_test.calculate_sharpe_ratio()
        sharpe_pred = portfolio_pred.calculate_sharpe_ratio()
    
        sharpe_ratio = pd.DataFrame({symbol+" Pred Sharpe Ratio":[sharpe_pred],symbol+" Test Sharpe Ratio: ":[sharpe_test]})
        annual_return = pd.DataFrame({symbol+" Pred Annualized Return":[portfolio_pred.total_return], symbol+" Test Annualized Return: ":[portfolio_test.total_return]})
        
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
        if not os.path.exists(os.path.join('data', 'rbm_random_forest',symbol,'Cumlative_P&L_Plot')):
                os.makedirs(os.path.join('data', 'rbm_random_forest',symbol,'Cumlative_P&L_Plot'))
        plt.savefig(os.path.join( 'data', 'rbm_random_forest',symbol,'Cumlative_P&L_Plot','Cum_P&L_'+symbol+'_'+str(i)+'.png'))
        plt.close()
        
        try:
            pred_port['Cumulative P&L'].plot(label='Predict', loglog = True)
            test_port['Cumulative P&L'].plot(label='Test', loglog = True)
            plt.gcf().autofmt_xdate()
            plt.ylabel("Value ($)")
            plt.xlabel("TIme")
            plt.legend()
            plt.title(symbol+'_'+str(i)+" Log Cumulative P&L")
            if not os.path.exists(os.path.join('data', 'rbm_random_forest',symbol,'Log_Cumlative_P&L_Plot')):
                os.makedirs(os.path.join('data', 'rbm_random_forest',symbol,'Log_Cumlative_P&L_Plot'))
            plt.savefig(os.path.join( 'data', 'rbm_random_forest',symbol,'Log_Cumlative_P&L_Plot','Log_Cum_P&L_'+symbol+'_'+str(i)+'.png'))
            plt.close()
        except ValueError:
            print symbol+'_'+str(i) +" no positive value"
            
        
        
        
        
        signal_pd['price'].plot()
        plt.gcf().autofmt_xdate()
        plt.ylabel("Value ($)")
        plt.xlabel("TIme")
        plt.legend()
        plt.title(symbol+'_'+str(i)+" Prices")
        plt.savefig(os.path.join( 'data', 'rbm_random_forest',symbol,'Prices_'+symbol+'_'+str(i)+'.png'))
        plt.close()
        
    s_f1_score1 = trans_mean_std(f1_score1_pd,symbol)
    s_f1_score2 = trans_mean_std(f1_score2_pd,symbol)
    s_classReport = trans_mean_std(ClassReport_pd,symbol)
    s_sharpeRatio = trans_mean_std(SharpeRatio_pd, symbol)
    s_return = trans_mean_std(AnnualReturn_pd, symbol)
    return s_f1_score1,s_f1_score2,s_classReport,s_sharpeRatio,s_return
        # chart
        # calculate sharpe ratio
        # compare with real signals


if __name__ == '__main__':
    capital = 100000
    f1_score1 = pd.DataFrame()
    f1_score2 = pd.DataFrame()
    classReport = pd.DataFrame()
    sharpeRatio = pd.DataFrame()
    annual_return = pd.DataFrame()
    
    contract = ut.read_data_from_file(os.path.join( 'data','rbm_random_forest', 'contract.csv'))
    for i in range(len(contract.index)):
        print contract.index[i]
        f1_score1_i, f1_score2_i, classReport_i, sharpeRatio_i, annual_return_i=back_testing_portfolio(contract.index[i], capital, float(contract['initial margin'].iloc[i]),float(contract['maint margin'].iloc[i]),float(contract['contract size'].iloc[i]), purchase_size = 1)
        f1_score1 = f1_score1.append(f1_score1_i)
        f1_score2 = f1_score2.append(f1_score2_i)
        classReport  = classReport.append(classReport_i)
        sharpeRatio  = sharpeRatio.append(sharpeRatio_i)
        annual_return  = annual_return.append(annual_return_i)
    
    f1_score1.to_csv(os.path.join( 'data','rbm_random_forest', 'f1_score_report_1.csv'))
    f1_score2.to_csv(os.path.join( 'data','rbm_random_forest', 'f1_score_report_2.csv'))
    classReport.to_csv(os.path.join( 'data','rbm_random_forest', 'Classification_error.csv'))
    sharpeRatio.to_csv(os.path.join( 'data','rbm_random_forest', 'Sharpe_Ratio.csv'))
    annual_return.to_csv(os.path.join( 'data','rbm_random_forest', 'Annulized_Return.csv'))

