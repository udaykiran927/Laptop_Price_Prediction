import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

file_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "data", "laptop_details.csv"))
file_path1 = os.path.abspath(os.path.join(os.getcwd(), "resources", "data", "laptop-types.csv"))
model_path = os.path.abspath(os.path.join(os.getcwd(), "resources", "models", "rfmodel.pkl"))
image_path=os.path.abspath(os.path.join(os.getcwd(), "resources", "data", "laptop.jpeg"))

df=pd.read_csv(file_path)
dff=pd.read_csv(file_path1)

with open(model_path, 'rb') as f:
    rf_model = pickle.load(f)

st.title('Flipkart Laptops Price Prediction')

add_selectbox = st.sidebar.selectbox(
    "How would you like to contact me?",
    ("linkedin", "Email", "Mobile phone")
)
d={"linkedin":'https://www.linkedin.com/in/uday-kiran-4aa25b1b5/','Email':'udaykirannaidu18@gmail.com',"Mobile phone":'+91-8074197277'}
k=d[add_selectbox]
st.write(k)


from PIL import Image
image = Image.open(image_path)
st.image(image)

# Get user input
brand = st.selectbox('Brand', dff['Brand'].unique())
os_options = dff.loc[dff['Brand'] == brand, 'OS'].unique()
os = st.selectbox('Operating System', os_options)
processor_options=dff.loc[dff['Brand']==brand, 'Processor'].unique()
processor = st.selectbox('Processor', processor_options)
print(processor)
display=st.selectbox('Display Length (cm)',dff['Display'].unique())
ram_type = st.selectbox('RAM Type', dff['RAMType'].unique())
ram_sizes = np.sort(dff['RAMSize'].unique())
ram_size = st.selectbox('RAM Size (in GB)', ram_sizes)
ssd_sizes = np.sort(dff['Storage_SSD'].unique())
ssd_size = st.selectbox('SSD Size (in GB)', ssd_sizes)
hdd_sizes = np.sort(dff['Storage_HDD'].unique())
hdd_size = st.selectbox('HDD Size (in GB)', hdd_sizes)

if st.button('Predict'):
    le_pr=LabelEncoder()
    dff['Processor']=le_pr.fit_transform(dff['Processor'])
    le_os = LabelEncoder()
    dff['OS']=le_os.fit_transform(dff['OS'])
    le_brand=LabelEncoder()
    dff['Brand']=le_brand.fit_transform(dff['Brand'])
    le_ramtype=LabelEncoder()
    dff['RAMType']=le_ramtype.fit_transform(dff['RAMType'])
    proc_enc = le_pr.transform([processor])[0]
    print(proc_enc)
    os_enc = le_os.transform([os])[0]
    b_enc = le_brand.transform([brand])[0]
    ram_enc = le_ramtype.transform([ram_type])[0]
    features = [b_enc,proc_enc,display,os_enc,ram_size,ram_enc,hdd_size, ssd_size]
    final_features = np.array(features).reshape(1, -1)
    prediction = rf_model.predict(final_features)
    
if 'prediction' not in locals():
    st.write('Click the "Predict" button to view estimated price.')    
else:
    # Display the prediction
    st.subheader('Prediction')  
    st.balloons()  
    st.write(f'The estimated price of the laptop is ₹ {prediction[0]:,.0f}.')


