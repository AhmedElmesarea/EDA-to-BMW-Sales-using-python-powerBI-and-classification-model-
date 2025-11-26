import numpy as np
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request
import pickle

# تحميل الموديل و encoders
model = pickle.load(open(r"D:\projects push to github\BMW sales\EDA_and_classification_ML_svm99_python\model.pkl", "rb")) 
le_model = pickle.load(open(r"D:\projects push to github\BMW sales\EDA_and_classification_ML_svm99_python\le_model.pkl", 'rb'))
le_region = pickle.load(open(r'D:\projects push to github\BMW sales\EDA_and_classification_ML_svm99_python\le_region.pkl', 'rb'))
le_color = pickle.load(open(r'D:\projects push to github\BMW sales\EDA_and_classification_ML_svm99_python\le_color.pkl', 'rb'))
le_fuel = pickle.load(open(r'D:\projects push to github\BMW sales\EDA_and_classification_ML_svm99_python\le_fuel.pkl', 'rb'))
le_trans = pickle.load(open(r'D:\projects push to github\BMW sales\EDA_and_classification_ML_svm99_python\le_trans.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None  # القيمة الافتراضية للنتيجة

    if request.method == 'POST':
        # جلب البيانات من الفورم
        model_input = request.form['model']
        year = int(request.form['year'])
        region_input = request.form['region']
        color_input = request.form['color']
        fuel_input = request.form['fuel_type']
        trans_input = request.form['transmission']
        engine_size = float(request.form['engine_size_l'])
        mileage = int(request.form['mileagw_km'])
        price = float(request.form['price_usd'])
        sales_volume = int(request.form['sales_volume'])

        # تحويل النصوص لأرقام باستخدام الـ encoders
        model_encoded = le_model.transform([model_input])[0]
        region_encoded = le_region.transform([region_input])[0]
        color_encoded = le_color.transform([color_input])[0]
        fuel_encoded = le_fuel.transform([fuel_input])[0]
        trans_encoded = le_trans.transform([trans_input])[0]

        # تكوين صف البيانات للموديل
        X = np.array([model_encoded, year, region_encoded, color_encoded,
                      fuel_encoded, trans_encoded, engine_size, mileage, price, sales_volume]).reshape(1, -1)

        # التنبؤ
        prediction = model.predict(X)[0]

        # تحويل الرقم لنص
        if prediction == 1:
            prediction_text = "High Sales"
        else:
            prediction_text = "Low Sales"

    # عرض الصفحة مع النتيجة (لو موجودة)
    return render_template('index.html', prediction=prediction_text)


if __name__ == '__main__':
    app.run(debug=True)
