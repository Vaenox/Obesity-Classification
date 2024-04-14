import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.header("Vücut Kilo Oranı Tespit Ekranı")
st.subheader("Belirtilen Değerleri Giriniz:",divider="red")

obesity = pd.read_csv(r"C:\Users\burak\Desktop\4.Sınıf Projeler\Serhat\Vize\Obezite Sınıflandırma\Obesity Classification.csv")

obesity["Gender"] = obesity["Gender"].apply(lambda x: 1 if x == "Male" else 0)

obesity["Label"] = obesity["Label"].replace("Underweight", 0)
obesity["Label"] = obesity["Label"].replace("Normal Weight", 1)
obesity["Label"] = obesity["Label"].replace("Overweight", 2)
obesity["Label"] = obesity["Label"].replace("Obese", 3)

X = obesity.drop(["Label", "ID"], axis=1)
y = obesity["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Model Kurma
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

model_svc = SVC(random_state=42)
model_svc.fit(X_train, y_train)

model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)

yas = st.number_input("Yaşınızı Giriniz:", min_value=1, max_value=112, step=1)
cinsiyet = st.number_input("Cinsiyetinizi Giriniz(Erkek=1 Kadın=0):", min_value=0, max_value=1, step=1)
boy = st.number_input("Boyunuzu Girin(cm):", min_value=0, max_value=210, step=1)
kilo = st.number_input("Kilonuzu Girin:", min_value=0, max_value=120, step=1)
bmi = st.number_input("BMI Endeksinizi Girin:", min_value=0, max_value=38, step=1)

if st.button("Random Forest ile Sonucu Hesapla"):
    pred_rf = model_rf.predict([[yas, cinsiyet, boy, kilo, bmi]])
    if pred_rf == 0:
        st.success("Test Sonucunuz: Zayıf")
    elif pred_rf == 1:
        st.info("Test Sonucunuz: Normal Kilo")
    elif pred_rf == 2:
        st.warning("Test Sonucunuz: Kilolu")
    elif pred_rf == 3:
        st.error("Test Sonucunuz: Obez")

if st.button("Support Vector Machine ile Sonucu Hesapla"):
    pred_svc = model_svc.predict([[yas, cinsiyet, boy, kilo, bmi]])
    if pred_svc == 0:
        st.success("Test Sonucunuz: Zayıf")
    elif pred_svc == 1:
        st.info("Test Sonucunuz: Normal Kilo")
    elif pred_svc == 2:
        st.warning("Test Sonucunuz: Kilolu")
    elif pred_svc == 3:
        st.error("Test Sonucunuz: Obez")

if st.button("K-Nearest Neighbors ile Sonucu Hesapla"):
    pred_knn = model_knn.predict([[yas, cinsiyet, boy, kilo, bmi]])
    if pred_knn == 0:
        st.success("Test Sonucunuz: Zayıf")
    elif pred_knn == 1:
        st.info("Test Sonucunuz: Normal Kilo")
    elif pred_knn == 2:
        st.warning("Test Sonucunuz: Kilolu")
    elif pred_knn == 3:
        st.error("Test Sonucunuz: Obez")

if st.button("Decision Tree ile Sonucu Hesapla"):
    pred_dt = model_dt.predict([[yas, cinsiyet, boy, kilo, bmi]])
    if pred_dt == 0:
        st.success("Test Sonucunuz: Zayıf")
    elif pred_dt == 1:
        st.info("Test Sonucunuz: Normal Kilo")
    elif pred_dt == 2:
        st.warning("Test Sonucunuz: Kilolu")
    elif pred_dt == 3:
        st.error("Test Sonucunuz: Obez")
