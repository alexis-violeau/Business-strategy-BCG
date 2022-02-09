import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np

FEATURES = ['mean_sales', 'mean_quantity','mean_product_per_order', 'n_orders', 'average_return_time', 'duration',
            'last_order_month', 'sales_net', 'quantity', 'product_id','time_since_last_command']
TARGET = ['churn']


def preprocess_data(df):
    '''Preprocessing transaction dataset'''
    
    #Aggregate orders by order date
    df_client_order = df.groupby(['client_id','date_order']).agg({'sales_net':'mean','quantity':'mean','product_id':'count'}).sort_index()
    df_client_order = df_client_order.reset_index()
    
    df_client_order['time_since_last_command'] = (df_client_order['date_order'] - df_client_order.groupby('client_id').shift(1)['date_order']).dt.days #Compute time between commands
    
    #Compute features at the client level
    df_client = df_client_order.groupby('client_id').agg(last_order = ('date_order','max'),
                                                     first_order = ('date_order','min'),
                                                     mean_sales = ('sales_net','mean'),
                                                     mean_quantity = ('quantity','mean'),
                                                     mean_product_per_order = ('product_id','mean'),
                                                     n_orders = ('product_id','count'),
                                                     average_return_time = ('time_since_last_command','mean')).reset_index().dropna()
    df_client['duration'] = (df_client['last_order'] - df_client['first_order']).dt.days
    df_client['last_order_month'] = df_client['last_order'].dt.month

    #We get the information from the last order
    return df_client.merge(df_client_order, right_on = ['client_id','date_order'], left_on = ['client_id','last_order'])
    
    
    
def split(data):
    '''Train test split on the preprocessed data'''
    #Identify churners in the dataset
    date_lim = datetime.strptime('01/01/2019', "%d/%m/%Y")
    data['churn'] = data['last_order'] < date_lim
    
    X_train, X_val, y_train, y_val = train_test_split(data[FEATURES],data[TARGET],stratify = data[TARGET])
    
    return X_train, X_val, y_train, y_val

