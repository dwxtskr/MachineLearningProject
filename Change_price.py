import csv
import glob
import numpy as np
import pandas as pd
import models.rbm_random_forest as rbm
import os
import sys
import datetime as dt
import math

params = dict(
n_row = 50000,
batchsize = 10,
learning_rate = 0.001,
n_iter = 50,
frac_train = 0.5,
frac_test = 0.25,
increment =1000,
n_symbol = 1,  # original value = 43
reduced_feature = 500,
n_estimator = 100,
criterion = 'entropy'
)

def read_price_from_csv(symbol, i,bottom=True):
    prices = []
    filename = os.path.join('data','csv', symbol +'*.csv')
    filenames = glob.glob(filename)

    if len(filenames) >= 1:
        filename = filenames[0]
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
            rows = list(reader)
            print len(rows)
            if bottom:
                start = int(-params['n_row']*(1-params['frac_train'])+params['increment']*i-1)
                prices = np.array(rows[:244580][start:start+int(params['n_row']*params['frac_test'])])
                print start, start+int(params['n_row']*params['frac_test'])
            else:
                start = int(params['n_row']*params ['frac_train'])+params['increment']*i+1
                prices = np.array(rows[start:start+int(params['n_row']*params['frac_test'])])
                print start, start+int(params['n_row']*params['frac_test'])
            prices = np.transpose(prices)

    return prices

def read_data_from_file(filename):
    file = glob.glob(filename)
    if len(file) >= 1:
        return pd.read_csv(file[0], index_col=0)
    
def convert_to_time(timestamp):
    temp = dt.timedelta(timestamp)
    time = dt.datetime(0001, 1, 1) + dt.timedelta(days=temp.days) + dt.timedelta(seconds=temp.seconds)
    # time = tm.gmtime(dt.timedelta(timestamp).seconds)
    #return time.strftime("%Y/%m/%d %H:%M:%S.%f")
    return time
    
def generate_data(symbol, iteration):
    filename = os.path.join('data','rbm_random_forest',symbol, 'price_'+symbol+'_'+str(iteration)+'.csv')
    file = glob.glob(filename)
    if len(file) < 1:
        csvname = os.path.join( 'data','rbm_random_forest', symbol, symbol+'_'+str(iteration)+'.csv')
        c_file = glob.glob(csvname)
        if len(c_file) < 1:
            path = os.path.join('data', 'smallBinaryPrice',symbol+'*')
            rbm.process_machine_learning(symbol, iteration, path)
        signal_pd = read_data_from_file(csvname)
        prices = read_price_from_csv(symbol, iteration)
        signal_pd['price']=prices[1]
        signal_pd.index = [convert_to_time(d) for d in prices[0]]
        signal_pd.to_csv(filename)
    return read_data_from_file(filename)

    

if __name__ == '__main__':
    folders = os.path.join('data','NotYetTest','*')
    folders= glob.glob(folders)
    for f_path in folders:
        symbol=f_path[-2:]
        for i in range(10):
            if not os.path.exists(os.path.join('data','NotYetTest',symbol)):
                os.makedirs(os.path.join('data','NotYetTest',symbol))
            filename = os.path.join('data','NotYetTest',symbol, 'price_'+symbol+'_'+str(i)+'.csv')
            file = glob.glob(filename)
            signal_pd = read_data_from_file(filename)
            signal_pd.columns = ['price', 'y_pred', 'y_test']
            signal_pd.to_csv(os.path.join('data','NotYetTest',symbol, 'price_'+symbol+'_'+str(i)+'.csv'))

