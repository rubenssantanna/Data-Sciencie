requirements.txt
from re import X
import streamlit as st 
from PIL import Image
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import glob
import os
import warnings
warnings.filterwarnings('ignore')

image = Image.open('heart.gif')
st.image(image, caption='App que usa Machine Learning para prever doença do coração, preencha os campos com as informações do paciente', width= 'auto', use_column_width= "auto")

# Importar Datframe
df = pd.read_csv('heart-statlog.csv')
df.replace({'absent': 0, 'present': 1},inplace=True)

# Cabeçalho
st.header("Prevendo Doenças do Coração")

# Nome do Paciente
name = st.sidebar.text_input("Digite o seu Nome")
st.write("PACIENTE:", name)

# Dados de Entrada
X= df.drop(columns=['class'] )
Y= df["class"]

def get_user_data():
    age= st.sidebar.slider ('Idade', 20,100, 60)
    sex = st.sidebar.selectbox('Sexo: 0= Masculino 1= Feminino', options= [0,1])
    chest= st.sidebar.selectbox('Dor no Peito: 1= nenhuma 2= Fraca 3= media 4=forte', options= [1,2,3,4])
    resting_blood_pressure = st.sidebar.slider('Pressão arterial sistólica em repouso', 100, 200, 120)
    serum_cholestoral= st.sidebar.slider('Colesterol sérico em mg / dl', 100, 500, 130 )
    fasting_blood_sugar_gt_120= st.sidebar.selectbox('Açúcar no sangue em jejum: 0<120 1>120',options= [0,1] )
    resting_electrocardiographic_re=st.sidebar.selectbox('Resultados eletrocardiográficos de repouso: 0= Normal 1= Moderado 2= Grave', options= [0,1,2])
    maximum_heart_rate_achieved=st.sidebar.slider('Frequência cardíaca máxima atingida batimentos por minuto', 100,200,140),
    exercise_induced_angina=  st.sidebar.selectbox('Faz Exercicio? 0= Nao 1 = Sim ', options= [0,1])
    oldpeak=st.sidebar.slider(' Medida de anormalidade nos eletrocardiogramas', 0.0, 5.0, 2.0)
    slope= st.sidebar.slider ('Fluxo sanguíneo no coração', 0, 10, 2)
    num_major_vessels= st.sidebar.selectbox('Número de vasos principais (0-3) coloridos por flourosopy', options= [0,1,2,3])
    thal=st.sidebar.selectbox('Resultados do teste de estresse com tálio que mede o fluxo sanguíneo para o coração: 3= valores normal 6= defeito fixo 7= defeito reversível', options= [3,6,7])
    
    user_data= { 'Idade': age,
                'Sexo: 0= Masculino 1= Feminino': sex,
                'Dor no Peito: 1= nemhuma 2= Fraca 3= media 4= forte': chest,
                'Pressão arterial sistólica em repouso': resting_blood_pressure,
                'Colesterol sérico em mg / dl':serum_cholestoral,
                'Açúcar no sangue em jejum: 0<120 1>120': fasting_blood_sugar_gt_120,
                'Resultados eletrocardiográficos de repouso: 0= Normal 1= Atenção 2= Grave': resting_electrocardiographic_re,
                'Frequência cardíaca máxima atingida batimentos por minuto': maximum_heart_rate_achieved,
                'Faz Exercicio? 0= Nao 1= Sim ': exercise_induced_angina,
                'Medida de anormalidade nos eletrocardiogramas': oldpeak,
                'Fluxo sanguíneo no coração': slope,
                'Número de vasos principais (0-3) coloridos por flourosopy':num_major_vessels,
                'Resultados do teste de estresse com tálio que mede o fluxo sanguíneo para o coração: 3= valores normal 6= defeito fixo 7= defeito reversível': thal          
                
               }
    
    
    
    features= pd.DataFrame(user_data, index=[0])
    return features

user_input_variables= get_user_data()

grafico=st.bar_chart(user_input_variables)

st.subheader('Dados do Paciente')
st.write(user_input_variables)

# Separa treino e teste
X_train, X_test, y_train, y_test= train_test_split(X,Y,test_size=.2, random_state=42)

# Método Random Forest Classifier
clf_RF = RandomForestClassifier() 
clf_RF.fit(X_train, y_train)  

# Acuracia do Modelo
y_pred = clf_RF.predict(X_test)
st.subheader('Acurácia do Modelo')
st.write(accuracy_score(y_test,clf_RF.predict(X_test))*100)
st.write("F1 Score: {}".format(f1_score(y_test, y_pred)))

# Previsão
st.subheader('Previsão:')
prediction= clf_RF.predict(user_input_variables)
st.write(prediction)
