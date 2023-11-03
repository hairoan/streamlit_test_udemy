import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
import plotly.express as px
import numpy as np
from PIL import Image
import yfinance as yf

def parameter(df, sector_default_val, cap_default_val):

    #sector
    sector_values = [sector_default_val] + list(df['sector'].unique())
    option_sector = st.sidebar.selectbox("Sector", sector_values, index = 0)

    #Market capitalization
    cap_value_list = [cap_default_val] + ['Small', 'Medium', 'Large']
    cap_value = st.sidebar.selectbox("Capitalisation", cap_value_list, index = 0)

    #dividend
    dividend_value = st.sidebar.slider('Dividend rate between than (%)', 0.0, 10.0, value= (0.0, 10.0))

    #Profit
    min_profit_value, max_profit_value = float(df['profitMargins_%'].min()), float(df['profitMargins_%'].max() )
    profit_value = st.sidebar.slider('Profit margin rate grater than (%) : ', min_profit_value, max_profit_value, step = 10.0)

    return option_sector, cap_value, dividend_value, profit_value

def filtering(df, sector_default_val, cap_default_val, option_sector, cap_value, dividend_value, profit_value):
    
    #Sector Filtering
    if option_sector != sector_default_val:
        df = df[df['sector']==option_sector]
    
    #Market capitalization filtering
    if  cap_value != cap_default_val:
        if cap_value == 'Small':
            df = df[ (df['marketCap'] >= 0)
                &
                (df['marketCap'] <= 20e9)]
            
        elif cap_value == 'Medium':
            df = df[ (df['marketCap'] >= 20e9)
                &
                (df['marketCap'] <= 100e9)]
        
        elif cap_value == 'Large':
            df = df[df['marketCap'] > 100e9]
    
    #dividend
    df = df[
        (df['dividendYield_%'] >= dividend_value[0])
        &
        (df['dividendYield_%'] <= dividend_value[1])
    ]

    #Profit
    df = df[df['profitMargins_%'] >= profit_value]

    return df
               

@st.cache(suppress_st_warning=True)

def read_data():
    my_path = "s&p500.csv"
    df = pd.read_csv(my_path)
    return df


def company_price(df,option_company):
    if option_company != None:
        ticker_company = df.loc[df['name'] == option_company, 'ticker'].values[0]
        data_price = pdr.get_data_yahoo(ticker_company, start="2011-12-31", end="2021-12-31")['Adj Close']
        data_price = data_price.reset_index(drop = False)
        data_price.columns = ['ds', 'y']
        return data_price
    
    return None

def show_stock_price(data_price):
    fig=px.line(data_price, x='ds', y='y', title = '10 years stock price')
    fig.update_xaxes(title_text = 'Date')
    fig.update_xaxes(title_text = 'Stock price')
    st.plotly_chart(fig)

def metrics(data_price):
    stock_price_2012 = data_price['y'].values[0]
    stock_price_2022 = data_price['y'].values[-1]
    performance = np.around((stock_price_2022/stock_price_2012 - 1)*100,2)
    return stock_price_2022,performance







if __name__ == '__main__':
    st.set_page_config(
        page_title= "udemy_project",
        page_icon = "📈",
        initial_sidebar_state = "expanded"
    )
    yf.pdr_override()
    st.title("S&P500 Screener & Analysis")
    st.sidebar.title("Search criteria")

    image = Image.open('stock.jpeg')
    _,col_image_2,_ =st.columns([1,3,1])
    with col_image_2:
        st.image(image, caption='@austindistel')

    df = read_data()

    sector_default_val = 'All'
    cap_default_val = 'All'
    option_sector, cap_value, dividend_value, profit_value = parameter(df, sector_default_val, cap_default_val)
    df = filtering(df, sector_default_val, cap_default_val, option_sector, cap_value, dividend_value, profit_value)
    
    
    st.subheader("Part 1 - S&P500 Screener")
    with st.expander("Part 1 explanation", expanded = False):
        st.write("""
            In the table below, you will find most of the companies in the S&P500 (stock market index of the 500 largest American companies) with certain criteria such as :
                
                - The name of the company
                - The sector of activity
                - Market capitalization
                - Dividend payout percentage (dividend/stock price)
                - The company's profit margin in percentage
            
            ⚠️ This data is scrapped in real time from the yahoo finance API. ⚠️

            ℹ️ You can filter / search for a company with the filters on the left. ℹ️
        """)
    st.write('Number of companies found : ', len(df))
    st.dataframe(df.iloc[:,1:])

    ## part2 - Choose a compagny
    st.subheader("Part 2 - Choose a company")
    option_company = st.selectbox("Choose a company : ", df.name.unique())

    ## Part 3 - Stock analysis

    st.subheader("Part 3 - {} Stock Analysis".format(option_company))
    data_price = company_price(df, option_company)
    

    # Stock price
    show_stock_price(data_price)

    stock_price_2022, performance = metrics(data_price)

    col_prediction_1, col_prediction_2 = st.columns([1,2])
    with col_prediction_1:
        st.metric(label="stock_price end 2022", value=str(np.around(stock_price_2022)), delta = str(performance)+ '%')
        st.write('* Compared to 3 jan. 2012')
    with col_prediction_2:
        with st.expander("Analysis explanatio", expanded = True):
            st.write("""
                The graph above shows the evolution of the selected stock price between 31st dec. 2011 and 31 dec. 2021.
                The indicator on the left is the stock price value in 31st dec. 2021 for the selected company and its evolution between 31st dec. 2011 and 31st dec. 2021.
                
                ⚠️⚠️ Theses value are computed based on what the Yahoo Finance API returns !
            """)

    
