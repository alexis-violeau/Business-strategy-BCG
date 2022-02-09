import streamlit as st 
from src import loading, metrics, modeling
import plotly.express as px
import pandas as pd

st.title('Sales dashboard')

client = st.sidebar.text_input('Client ID',188502)

@st.cache
def load():
    return loading.load_data(), modeling.load_model()
    
df,model = load()
df['date_order'] = pd.to_datetime(df['date_order'] )

df_client = df[df['client_id'] == int(client)]


df_client_order = df_client[['date_order','quantity','sales_net']].groupby('date_order').sum().reset_index().sort_values(by = 'date_order',ascending = False)
mean =  df_client_order.agg({'quantity' : 'mean', 'sales_net' : ['mean','count']})
last_order = df_client_order.head(1)


quantity = df_client_order.iloc[0,1]
sales = df_client_order.iloc[0,2]
mean_sales = mean['sales_net']['mean']
mean_quantity = mean['quantity']['mean']
n_orders = mean['sales_net']['count']
days_since_last_order = (df_client_order.iloc[0,0] - df_client_order.iloc[1,0]).days
days_client = (df_client_order['date_order'].max() - df_client_order['date_order'].min()).days
month = df_client_order.iloc[0,0].month

churn_probability = model.predict_proba([[sales,quantity,mean_sales,mean_quantity,n_orders,days_since_last_order,days_client,month]])[0][1]
lifetime_value = metrics.lifetime_value(df_client,client = int(client))

st.sidebar.metric(label = 'Churn probability', value = '{} %'.format(100*round(churn_probability,3)))

if lifetime_value > 10e9:
    st.sidebar.metric(label = 'Lifetime value', value = '{} B $'.format(round(lifetime_value/10e9,2)))
elif lifetime_value > 10e6:
    st.sidebar.metric(label = 'Lifetime value', value = '{} M $'.format(round(lifetime_value/10e6,2)))
elif lifetime_value > 10e3:
    st.sidebar.metric(label = 'Lifetime value', value = '{} k $'.format(round(lifetime_value/10e3,2)))
else:
    st.sidebar.metric(label = 'Lifetime value', value = '{} $'.format(round(lifetime_value,2)))
    
if df_client_order.iloc[0,2] > 10e9:
    st.sidebar.metric(label = 'Last sale', value = '{} B $'.format(round(df_client_order.iloc[0,2]/10e9,2)),delta = '{} B $'.format(round((df_client_order.iloc[0,2] - df_client_order.iloc[1,2])/10e9,2)))
elif df_client_order.iloc[0,2] > 10e6:
    st.sidebar.metric(label = 'Last sale', value = '{} M $'.format(round(df_client_order.iloc[0,2]/10e6,2)),delta = '{} M $'.format(round((df_client_order.iloc[0,2] - df_client_order.iloc[1,2])/10e6,2)/10e6))
elif df_client_order.iloc[0,2] > 10e3:
    st.sidebar.metric(label = 'Last sale', value = '{} k $'.format(round(df_client_order.iloc[0,2]/10e3,2)),delta = '{} k $'.format(round((df_client_order.iloc[0,2] - df_client_order.iloc[1,2])/10e3,2)))
else:
    st.sidebar.metric(label = 'Last sale', value = '{} $'.format(round(df_client_order.iloc[0,2],2)),delta = '{} $'.format(round(df_client_order.iloc[0,2] - df_client_order.iloc[1,2],2)))

st.sidebar.metric(label = 'Client relationship', value = '{} days'.format(days_client))


history = px.bar(df_client, x = 'date_order', y = 'sales_net', color = 'order_channel', title = 'Transaction history')
st.plotly_chart(history.update_layout(width=1000,height = 400))

pie_chart = px.pie(df_client, values = 'sales_net',names = 'order_channel',title = 'Order channels')
st.plotly_chart(pie_chart)

top_products = df_client[['sales_net','product_id']].groupby('product_id').sum().sort_values(by = 'sales_net',ascending = False).reset_index()
top_products['product_id'] = top_products['product_id'].astype(str)

bar_plot = px.bar(top_products.head(3), x="sales_net", y="product_id", orientation='h',title = 'Top products : ')
st.plotly_chart(bar_plot)



