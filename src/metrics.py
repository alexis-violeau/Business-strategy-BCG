def lifetime_value(df,client):
    df_client = df[df['client_id'] == client]
    duration = df_client['date_order'].max() - df_client['date_order'].min()
    total_sum = float(df_client['sales_net'].sum())
    
    r = 0.05
    duration_year = duration.days/365
    dcf = total_sum/duration_year * (1+r)/r
    
    return dcf


def gain(model,df,churn,promotion_ratio = 0.1):
    r = 0.005
    
    y_pred = model.predict(df)
    costs = y_pred * df['mean_sales'] * promotion_ratio
    rewards = y_pred * churn * df['mean_sales'] * df['n_orders'] / df['days_client'] * 365 * (1+r)/r
    
    return rewards - costs