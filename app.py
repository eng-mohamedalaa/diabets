import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# تحميل البيانات
csv_file_path = 'Diabetes Dataset_Training Part.csv'
data = pd.read_csv(csv_file_path)

# تقسيم البيانات إلى ميزات (X) وعلامات (y)
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحجيم البيانات
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# تدريب نموذج Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# حفظ النموذج المدرب والمقياس
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# دالة التنبؤ
def predict_diabetes(input_data):
    # تحميل النموذج المُدرب والمقياس
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # تحجيم البيانات المدخلة
    input_data_scaled = scaler.transform(input_data)
    
    # إجراء التنبؤ
    prediction = model.predict(input_data_scaled)
    
    # عرض النتيجة
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# واجهة المستخدم باستخدام Streamlit
def main():
    st.title("Diabetes Prediction App")
    
    # إدخالات المستخدم
    st.header("Input the patient's data")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=33)

    # زر التنبؤ
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'Preg': [pregnancies],
            'Glucose': [glucose],
            'BPressure': [blood_pressure],
            'SThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })
        
        outcome = predict_diabetes(input_data)
        st.subheader("Prediction Results")
        st.write(f"Outcome: {outcome}")

# تشغيل التطبيق
if __name__ == "__main__":
    main()
