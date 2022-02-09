import streamlit as st 
from src import loading, metrics, modeling, preprocessing
import plotly.express as px
import pandas as pd

st.title('Sales dashboard')

client = st.sidebar.text_input('Client ID',188502)

@st.cache
def load():
    return loading.load_data(), modeling.load_model()
    
df,model = load()

df_client = df[df['client_id'] == int(client)]
df_client_preprocess= preprocessing.preprocess_data(df_client)

churn_probability = model.predict_proba(df_client_preprocess[preprocessing.FEATURES])[0][1]
lifetime_value = metrics.lifetime_value(df_client_preprocess)

st.sidebar.metric(label = 'Churn probability', value = '{} %'.format(100*round(churn_probability,3)))
st.sidebar.metric(label = 'Lifetime value', value = '{} $'.format(round(lifetime_value,2)))
st.sidebar.metric(label = 'Last sale', value = '{} $'.format(round(df_client_preprocess['sales_net'][0] ,2)),delta = '{} $'.format(round(df_client_preprocess['sales_net'][0] - df_client_preprocess['mean_sales'][0],2)))
st.sidebar.metric(label = 'Client relationship', value = '{} days'.format(df_client_preprocess['duration'][0] ))

history = px.bar(df_client, x = 'date_order', y = 'sales_net', color = 'order_channel', title = 'Transaction history')
st.plotly_chart(history.update_layout(width=1000,height = 400))

pie_chart = px.pie(df_client, values = 'sales_net',names = 'order_channel',title = 'Order channels')
st.plotly_chart(pie_chart)

top_products = df_client[['sales_net','product_id']].groupby('product_id').sum().sort_values(by = 'sales_net',ascending = False).reset_index()
top_products['product_id'] = top_products['product_id'].astype(str)

bar_plot = px.bar(top_products.head(3), x="sales_net", y="product_id", orientation='h',title = 'Top products : ')
st.plotly_chart(bar_plot)



