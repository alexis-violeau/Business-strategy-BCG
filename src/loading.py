import pandas as pd


def load_data(path = 'data/transactions_dataset.csv'):

    df = pd.read_csv(path,sep = ';')
    df['date_order'] = pd.to_datetime(df['date_order'])
    df['date_invoice'] = pd.to_datetime(df['date_invoice'])
    
    return df