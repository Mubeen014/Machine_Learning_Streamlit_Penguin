import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Penguins Prediction App

This app preditcs the penguin species.
         

""")

st.sidebar.header('User Inputs')

def user_inputs():
    island = st.sidebar.selectbox('Islands', ('Biscoe','Dream','Torgersen'))
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
    bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
    flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
    body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
    data = {
        'island': island,
        'sex': sex,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_inputs()

df_peng = pd.read_csv('penguins_cleaned.csv')
penguin_species = np.array(sorted(df_peng['species'].unique()))
df_peng.drop(columns=['species'], inplace=True)
df = pd.concat([input_df, df_peng], axis=0)

dummy = pd.get_dummies(df['island'])
df.drop(['island'], axis=1, inplace=True)
df = pd.concat([df, dummy], axis=1)

dummy = pd.get_dummies(df['sex'])
df.drop(['sex'], axis=1, inplace=True)
df = pd.concat([df, dummy], axis=1)

df_user = df[:1]

load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))
prediction = load_clf.predict(df_user)
prediction_proba = load_clf.predict_proba(df_user)

st.subheader('Prediction')
st.write(penguin_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)