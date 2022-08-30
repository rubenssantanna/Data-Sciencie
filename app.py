from re import X
import streamlit as st 
import pandas as pd
import numpy as np
df = pd.read_csv("credit_default.csv", index_col=0)
df.head()
df3= df.groupby("age")["income","term"].mean()
df1= df.groupby("loan_type")["age", "gender", "term"].mean()
df4= df.groupby("gender")["age", "credit_score", "status"].mean()
st.header("Dashboard Inadimplência de Crédito")
#if st.sidebar.button("Escolha o Grafico"):
    #df3= pd.DataFrame(
     #   np.random.rand(20,3),
      #  columns=[ 'age','income', 'status']
    #)
    #st.bar_chart(df3)
    #st.line_chart(df3)
    
    
    
opcao= st.sidebar.multiselect(
    "SELECIONE A OPÇÃO",
    ('LOAN_TYPE','GENDER','AGE')
) 

if "LOAN_TYPE" in opcao:
    st.line_chart(df1)
if "GENDER" in opcao:
    st.bar_chart(df4)
if "AGE" in opcao:
    st.line_chart(df1)


        
        
        
        
        
        


