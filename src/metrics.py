def lifetime_value(df_client,r = 0.05):
    '''Discounted cash flow assuming constant sales per year'''
    return (365 * df_client['mean_sales'] * df_client['n_orders']/df_client['duration'] * (1+r)/r)[0]


def model_performance(model,df,churn,promotion_ratio = 0.1):
    r = 0.005
    
    y_pred = model.predict(df)
    costs = y_pred * df['mean_sales'] * promotion_ratio
    rewards = y_pred * churn * df['mean_sales'] * df['n_orders'] / df['days_client'] * 365 * (1+r)/r
    
    return rewards - costs