import pandas as pd
from datetime import datetime


def preprocess_data(df):

    df2 = df[['date_order','client_id','sales_net','quantity']].groupby(['client_id','date_order']).agg({'sales_net' : 'sum', 'quantity' : 'sum'}).reset_index().sort_values(['client_id','date_order'])

    old_client_id = -1
    mean_sales = []
    mean_quantity = []
    n_orders = []
    days_since_last_order = []
    days_client = []
    churn = []
    old_date = datetime.strptime('01/01/2017', "%d/%m/%Y")
    date_lim = datetime.strptime('01/01/2019', "%d/%m/%Y")

    for row in df2.itertuples():
        client_id = row[1]
        
        quantity = row[4]
        sales = row[3]
        date = row[2]
        
        
        if old_client_id == client_id:
            mean_sales.append((mean_sales[-1] * n_orders[-1] + sales)/(n_orders[-1] + 1))
            mean_quantity.append((mean_quantity[-1] * n_orders[-1] + quantity)/(n_orders[-1] + 1))
            n_orders.append(n_orders[-1]+1)
            days_since_last_order.append((date - old_date).days)
            days_client.append(days_client[-1] + days_since_last_order[-1])
            churn.append(0)
            
        else:
            mean_sales.append(sales)
            mean_quantity.append(quantity)
            n_orders.append(1)
            days_since_last_order.append(-1)
            days_client.append(0)
            
            if old_date < date_lim:
                churn.append(1)
            else:
                churn.append(-1)

        old_client_id = client_id
        old_date = date
        
    if old_date < date_lim:
        churn.append(1)
    else:
        churn.append(-1)
        
    df2['mean_sales'] = mean_sales
    df2['mean_quantity'] = mean_quantity
    df2['n_orders'] = n_orders
    df2['days_since_last_order'] = days_since_last_order
    df2['churn'] = churn[1:]
    df2['month'] = df2['date_order'].dt.month
    df2['days_client'] = days_client


    df_train = df2[df2['churn'] != -1]
    df_test = df2[df2['churn'] == -1]

    return df_train, df_test