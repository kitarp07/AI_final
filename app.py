import streamlit as st
import numpy as np

import pickle 


model = pickle.load(open("C:/Users/user/AI_final/model.sav", "rb"))


def predict_heart_disease(variables):

    input = np.asarray(variables)

    reshape_input = input.reshape(1, -1)

    result = model.predict(reshape_input)

    return result

def main():
    st.title("Heart disease prediction using machine learning model")

    age = st.text_input("Age", "Type here")
    sex = st.selectbox("Sex", ["Male", "Female"])
    cigsPerDay = st.text_input("Cigarettes per Day", "Type here")
    sysBP = st.text_input("Systolic blood pressure", "Type here")
    diaBP = st.text_input("Diastolic blood pressure", "Type here")
    BMI = st.text_input("Body Mass Index", "Type here")
    heartrate = st.text_input("Heart rate", "Type here")
    diabetes = st.selectbox("Has Diabetes?", ["Select option","Yes", "No"])
    prevalentHyp = st.selectbox("Has Prevalent Hypertension?", ["Select option","Yes", "No"])

    result = ""
    gender = 0
    isDiabetic =0
    PrevHyp = 0

    


    if st.button("Predict"):

        if sex == "Male":
            gender =1
        else:
            gender = 0

        if diabetes == "Yes":
            isDiabetic =1
        else:
            PrevHyp = 0

        if prevalentHyp == "Yes":
            PrevHyp =1
        else:
            PrevHyp = 0



        input = np.asarray([age, gender, cigsPerDay, sysBP, diaBP, BMI, heartrate, isDiabetic, PrevHyp])

        reshaped_input = input.reshape(1, -1)
        prediction = model.predict(reshaped_input)

        if (prediction[0]==0):
            result = "Person does not have heart disease"
        else:
            result = "Person has heart disease"
    st.success(result)

if __name__ == '__main__':
    main()
