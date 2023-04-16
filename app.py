# Import necessary packages
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
model=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))




page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://akm-img-a-in.tosshub.com/businesstoday/images/story/202301/air-india-1200-sixteen_nine.jpg");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

</style>
"""

st.markdown( page_bg_img, unsafe_allow_html=True)


new_title = '<p style="font-family:CODE Bold; color:black; font-size: 50px;text-align: center;">Air Passenger forecasts</p>'
# Create Streamlit app
st.markdown(new_title, unsafe_allow_html=True)
#st.title("<h1 style='color: red;'>Air Passengers forecast</h1>")
new_title1 = '<p style="font-family:Regular 400; color:blue; font-size: 30px;text-align: center;">This app uses SARIMA to forecast passenger traffic.</p>'
st.markdown(new_title1, unsafe_allow_html=True)
#t.write('This app uses SARIMA to forecast passenger traffic.')
#start_date = st.text_input('Start Date (YYYY-MM-DD):')

# Make prediction


# Add user inputs
start_date = st.date_input('Start date', value=pd.to_datetime('1960-01-01'))
end_date = st.date_input('End date', value=pd.to_datetime('1970-12-01'))

# Make predictions
pred = model.predict(start=start_date, end=end_date, dynamic=True)

# Plot results
st.line_chart(pred)
#st.write(pred)
fig, ax = plt.subplots(figsize=(10, 6))
df.plot(ax=ax)
pred.plot(ax=ax)
plt.legend(['Observed', 'Predicted'])
plt.title('Predicted Values')
plt.xlabel('Date')
plt.ylabel('Passengers')
st.pyplot(fig)


forecast_df = pd.DataFrame(pred)
forecast_df.index.name = 'date'
forecast_df.reset_index(inplace=True)
# Display the forecasted values as a table
st.table(forecast_df)
