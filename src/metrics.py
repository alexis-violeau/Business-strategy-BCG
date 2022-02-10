def lifetime_value(df_client,r = 0.02):
    '''Discounted cash flow assuming constant sales per year'''
    return (365 * df_client['mean_sales'] * df_client['n_orders']/df_client['duration'] * (1+r)/r)[0]


def model_performance(model,df,churn,promotion_ratio = 0.3):
    y_pred = model.predict(df)
    
    # When predicting churn, we offer a promotion corresponding to a share of its next order
    costs = y_pred * df['mean_sales'] * promotion_ratio
    
    # If we correctly predicted churn, our promotion makes him stay for at least one year
    rewards = (y_pred * churn) * df['mean_sales'] * df['n_orders'] / df['duration'] * 365
        
    return rewards - costs