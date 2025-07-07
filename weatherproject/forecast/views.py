from django.shortcuts import render
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import requests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
import matplotlib.pyplot as plt
import io
import base64

# 🔹 Utility: Get real-time weather
def get_current_weather(city):
    API_KEY = '5cb5519e0f137832e21b7a77ca27b51f'
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            raise Exception(data.get("message", "Failed to fetch weather"))
        return {
            'current_temp': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'temp_min': data['main']['temp_min'],
            'temp_max': data['main']['temp_max'],
            'humidity': data['main']['humidity'],
            'rain_last_hour': data.get('rain', {}).get('1h', 0),
            'rain_expected': np.random.uniform(0, 100),
            'cloudcover': data['clouds']['all'],
            'wind_gust': data['wind'].get('gust', 10),
            'wind_speed': data['wind']['speed'],
            'wind_dir': data['wind']['deg'],
            'sea_level': data['main'].get('sea_level', 1010),
            'visibility': data.get('visibility', 10000),
            'description': data['weather'][0]['description'],
            'country': data['sys']['country']
        }
    except Exception as e:
        print(f"❌ Error fetching weather: {e}")
        return None

# 🔹 Helpers
def prepare_data(df):
    features = ['temp', 'humidity', 'precip', 'precipprob', 'windgust', 'windspeed', 'winddir', 'sealevelpressure']
    X = df[features]
    le = LabelEncoder()
    y = le.fit_transform(df['description'])
    return X, y, le

def prepare_regression_data(df, target):
    if target == 'temp':
        X = df[['humidity', 'precipprob', 'windgust']]
    elif target == 'humidity':
        X = df[['temp', 'precipprob', 'windgust']]
    else:
        X = df[['temp', 'humidity', 'windgust']]
    y = df[target]
    return X, y

def predict_future(model, current_input, model_type):
    predictions = []
    current_df = current_input.copy()
    for _ in range(8):
        pred = model.predict(current_df)[0]
        if model_type == 'temp':
            pred = np.clip(pred, 10, 45)
        elif model_type == 'humidity':
            pred = np.clip(pred, 10, 100)
        elif model_type == 'precip':
            pred = np.clip(pred, 0, 100)
        predictions.append(pred)
        if model_type == 'temp':
            h = np.clip(current_df['humidity'].values[0] + np.random.uniform(-2, 2), 10, 100)
            p = np.clip(current_df['precipprob'].values[0] + np.random.uniform(-5, 5), 0, 100)
            current_df = pd.DataFrame([{
                'humidity': h,
                'precipprob': p,
                'windgust': current_df['windgust'].values[0]
            }])
        elif model_type == 'humidity':
            t = np.clip(current_df['temp'].values[0], 10, 45)
            p = np.clip(current_df['precipprob'].values[0] + np.random.uniform(-5, 5), 0, 100)
            current_df = pd.DataFrame([{
                'temp': t,
                'precipprob': p,
                'windgust': current_df['windgust'].values[0]
            }])
        else:
            t = np.clip(current_df['temp'].values[0], 10, 45)
            h = np.clip(current_df['humidity'].values[0], 10, 100)
            current_df = pd.DataFrame([{
                'temp': t,
                'humidity': h,
                'windgust': current_df['windgust'].values[0]
            }])
    return predictions

def generate_mock_data():
    return pd.DataFrame({
        'temp': np.random.uniform(15, 35, 100),
        'humidity': np.random.uniform(40, 90, 100),
        'precip': np.random.uniform(0, 5, 100),
        'precipprob': np.random.uniform(0, 100, 100),
        'windgust': np.random.uniform(5, 20, 100),
        'windspeed': np.random.uniform(2, 15, 100),
        'winddir': np.random.uniform(0, 360, 100),
        'sealevelpressure': np.random.uniform(1000, 1020, 100),
        'description': np.random.choice(
            ['clear sky', 'fewclouds', 'scatteredclouds', 'brokenclouds', 'shower rain', 'rain', 'thunderstorm', 'snow', 'mist'], 100
        )
    })

# 🔹 Main View
def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        current_weather = get_current_weather(city)
        if current_weather is None:
            return render(request, 'weather.html', {'error': "City not found"})

        historical_data = generate_mock_data()
        X, y, le = prepare_data(historical_data)
        rain_model = LogisticRegression(max_iter=1000).fit(X, y)

        current_df = pd.DataFrame([{
            'temp': current_weather['current_temp'],
            'humidity': current_weather['humidity'],
            'precip': current_weather['rain_last_hour'],
            'precipprob': current_weather['rain_expected'],
            'windgust': current_weather['wind_gust'],
            'windspeed': current_weather['wind_speed'],
            'winddir': current_weather['wind_dir'],
            'sealevelpressure': current_weather['sea_level'],
        }])
        label = rain_model.predict(current_df)[0]
        description_prediction = le.inverse_transform([label])[0]

        # Regression Models
        x_temp, y_temp = prepare_regression_data(historical_data, 'temp')
        x_hum, y_hum = prepare_regression_data(historical_data, 'humidity')
        x_precip, y_precip = prepare_regression_data(historical_data, 'precipprob')

        temp_model = LinearRegression().fit(x_temp, y_temp)
        hum_model = LinearRegression().fit(x_hum, y_hum)
        precip_model = LinearRegression().fit(x_precip, y_precip)

        temp_input = pd.DataFrame([{
            'humidity': current_weather['humidity'],
            'precipprob': current_weather['rain_expected'],
            'windgust': current_weather['wind_gust']
        }])
        hum_input = pd.DataFrame([{
            'temp': current_weather['current_temp'],
            'precipprob': current_weather['rain_expected'],
            'windgust': current_weather['wind_gust']
        }])
        precip_input = pd.DataFrame([{
            'temp': current_weather['current_temp'],
            'humidity': current_weather['humidity'],
            'windgust': current_weather['wind_gust']
        }])

        linear_forecast_temp = predict_future(temp_model, temp_input, 'temp')
        linear_forecast_humidity = predict_future(hum_model, hum_input, 'humidity')
        linear_forecast_precip = predict_future(precip_model, precip_input, 'precip')

        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        future_times = [(now + timedelta(hours=i+1)).strftime("%H:00") for i in range(8)]

        time1, time2, time3, time4, time5, time6, time7, time8 = future_times
        future_temp = linear_forecast_temp[:8]
        future_humidity = linear_forecast_humidity[:8]
        future_precip = linear_forecast_precip[:8]

        temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8 = future_temp
        humidity1, humidity2, humidity3, humidity4, humidity5, humidity6, humidity7, humidity8 = future_humidity
        precip1, precip2, precip3, precip4, precip5, precip6, precip7, precip8 = future_precip

        context = {
            'location': city,
            'tempmin': current_weather['temp_min'],
            'tempmax': current_weather['temp_max'],
            'temp': current_weather['current_temp'],
            'humidity': current_weather['humidity'],
            'windgust': current_weather['wind_gust'],
            'windspeed': current_weather['wind_speed'],
            'winddir': current_weather['wind_dir'],
            'sealevelpressure': current_weather['sea_level'],
            'feels_like': current_weather['feels_like'],
            'description': current_weather['description'],
            'country': current_weather['country'],
            'visibility': current_weather['visibility'],
            'description_prediction': description_prediction,
            'time': datetime.now(),
            'date': datetime.now().strftime("%B %d, %Y"),
            'time1': time1, 'time2': time2, 'time3': time3,
            'time4': time4, 'time5': time5, 'time6': time6,
            'time7': time7, 'time8': time8, 
            'temp1': f"{round(temp1, 1)}", 'temp2': f"{round(temp2, 1)}", 'temp3': f"{round(temp3, 1)}",
            'temp4': f"{round(temp4, 1)}", 'temp5': f"{round(temp5, 1)}", 'temp6': f"{round(temp6, 1)}",
            'temp7': f"{round(temp7, 1)}", 'temp8': f"{round(temp8, 1)}", 
            'humidity1': f"{round(humidity1, 1)}", 'humidity2': f"{round(humidity2, 1)}", 'humidity3': f"{round(humidity3, 1)}",
            'humidity4': f"{round(humidity4, 1)}", 'humidity5': f"{round(humidity5, 1)}", 'humidity6': f"{round(humidity6, 1)}",
            'humidity7': f"{round(humidity7, 1)}", 'humidity8': f"{round(humidity8, 1)}",
            'precip1': f"{round(precip1, 1)}", 'precip2': f"{round(precip2, 1)}", 'precip3': f"{round(precip3, 1)}",
            'precip4': f"{round(precip4, 1)}", 'precip5': f"{round(precip5, 1)}", 'precip6': f"{round(precip6, 1)}",
            'precip7': f"{round(precip7, 1)}", 'precip8': f"{round(precip8, 1)}",
        }
        return render(request, 'forecast\weather.html', context)
    return render(request, 'forecast\weather.html')


#........ARIMA MODEL........#
def arima_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        
        # Load historical data
        df = pd.read_csv('C:\\Users\\shreeyaa\\.vscode\\Newfolder\\weatherpersonalised\\asia_01_20.csv')  # ✅ Update this path
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.asfreq('H').fillna(method='ffill')

        # Fetch live data
        API_KEY = '5cb5519e0f137832e21b7a77ca27b51f'
        url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            return render(request, 'weather.html', {'error': f"City not found: {city}"})

        current = {
            'city': city.title(),
            'country': data['sys']['country'],
            'temp': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'temp_min': data['main']['temp_min'],
            'temp_max': data['main']['temp_max'],
            'humidity': data['main']['humidity'],
            'rain_last_hour': data.get('rain', {}).get('1h', 0),
            'precipprob': np.random.uniform(10, 90),
            'cloudcover': data['clouds']['all'],
            'windgust': data['wind'].get('gust', 0),
            'windspeed': data['wind']['speed'],
            'winddir': data['wind'].get('deg', 0),
            'visibility': data.get('visibility', 10000),
            'description': data['weather'][0]['description']
        }

        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        future_times = [(now + timedelta(hours=i+1)).strftime("%H:00") for i in range(8)]

        # 🔹 ARIMA for Temperature
        delta_temp = df['temp'].last('7D').diff().dropna()
        model_temp = ARIMA(delta_temp, order=(1, 0, 1)).fit()
        forecast_temp = current['temp'] + np.cumsum(model_temp.forecast(steps=8))
        forecast_temp = np.clip(forecast_temp, current['temp'] - 10, current['temp'] + 10)

        # 🔹 ARIMA for Humidity
        delta_hum = df['humidity'].last('7D').diff().dropna()
        model_hum = ARIMA(delta_hum, order=(1, 0, 1)).fit()
        forecast_hum = current['humidity'] + np.cumsum(model_hum.forecast(steps=8))
        forecast_hum = np.clip(forecast_hum, 10, 100)

        # 🔹 ARIMA for Precipitation Probability
        if 'precipprob' not in df.columns:
            trend = np.linspace(20, 80, len(df))
            noise = np.random.normal(0, 7, len(df))
            df['precipprob'] = np.clip(trend + noise, 0, 100)
        delta_precip = df['precipprob'].last('7D').diff().dropna()
        model_precip = ARIMA(delta_precip, order=(1, 0, 1)).fit()
        forecast_precip = current['precipprob'] + np.cumsum(model_precip.forecast(steps=8))
        forecast_precip = np.clip(forecast_precip, 0, 100)

        # Split for frontend
        temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8 = forecast_temp
        humidity1, humidity2, humidity3, humidity4, humidity5, humidity6, humidity7, humidity8 = forecast_hum
        precip1, precip2, precip3, precip4, precip5, precip6, precip7, precip8 = forecast_precip

        time1, time2, time3, time4, time5, time6, time7, time8 = future_times

        context = {
            'location': city,
            'tempmin': current['temp_min'],
            'tempmax': current['temp_max'],
            'temp': current['temp'],
            'humidity': current['humidity'],
            'windgust': current['windgust'],
            'windspeed': current['windspeed'],
            'winddir': current['winddir'],
            'sealevelpressure': 1010,  # Placeholder
            'feels_like': current['feels_like'],
            'description': current['description'],
            'country': current['country'],
            'visibility': current['visibility'],
            'time': datetime.now(),
            'date': datetime.now().strftime("%B %d, %Y"),

            'time1': time1, 'time2': time2, 'time3': time3,
            'time4': time4, 'time5': time5, 'time6': time6,
            'time7': time7, 'time8': time8,

            'temp1': f"{round(temp1, 1)}", 'temp2': f"{round(temp2, 1)}", 'temp3': f"{round(temp3, 1)}",
            'temp4': f"{round(temp4, 1)}", 'temp5': f"{round(temp5, 1)}", 'temp6': f"{round(temp6, 1)}",
            'temp7': f"{round(temp7, 1)}", 'temp8': f"{round(temp8, 1)}",

            'humidity1': f"{round(humidity1, 1)}", 'humidity2': f"{round(humidity2, 1)}",
            'humidity3': f"{round(humidity3, 1)}", 'humidity4': f"{round(humidity4, 1)}",
            'humidity5': f"{round(humidity5, 1)}", 'humidity6': f"{round(humidity6, 1)}",
            'humidity7': f"{round(humidity7, 1)}", 'humidity8': f"{round(humidity8, 1)}",

            'precip1': f"{round(precip1, 1)}", 'precip2': f"{round(precip2, 1)}",
            'precip3': f"{round(precip3, 1)}", 'precip4': f"{round(precip4, 1)}",
            'precip5': f"{round(precip5, 1)}", 'precip6': f"{round(precip6, 1)}",
            'precip7': f"{round(precip7, 1)}", 'precip8': f"{round(precip8, 1)}",
        }

        return render(request, 'forecast/arima.html', context)
    return render(request, 'forecast/arima.html')

#...........SARIMA MODEL...........#
def sarima_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')

        # Load data
        df = pd.read_csv('C:\\Users\\shreeyaa\\.vscode\\Newfolder\\weatherpersonalised\\asia_01_20.csv')  # ✅ Replace with correct path
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.asfreq('H').fillna(method='ffill')

        # Fetch current weather
        API_KEY = '5cb5519e0f137832e21b7a77ca27b51f'
        url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            return render(request, 'weather.html', {'error': f"City not found: {city}"})

        current = {
            'city': city.title(),
            'country': data['sys']['country'],
            'temp': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'temp_min': data['main']['temp_min'],
            'temp_max': data['main']['temp_max'],
            'humidity': data['main']['humidity'],
            'rain_last_hour': data.get('rain', {}).get('1h', 0),
            'precipprob': np.random.uniform(10, 90),
            'cloudcover': data['clouds']['all'],
            'windgust': data['wind'].get('gust', 0),
            'windspeed': data['wind']['speed'],
            'winddir': data['wind'].get('deg', 0),
            'visibility': data.get('visibility', 10000),
            'description': data['weather'][0]['description']
        }

        # Forecasts
        temp_model = SARIMAX(df['temp'].last('7D').diff().dropna(), order=(1, 0, 1), seasonal_order=(1, 0, 1, 24)).fit(disp=False)
        temp_delta = temp_model.forecast(steps=8)
        forecast_temp = current['temp'] + np.cumsum(temp_delta)
        forecast_temp = np.clip(forecast_temp, current['temp'] - 10, current['temp'] + 10)

        hum_model = SARIMAX(df['humidity'].last('7D').diff().dropna(), order=(1, 0, 1), seasonal_order=(1, 0, 1, 24)).fit(disp=False)
        hum_delta = hum_model.forecast(steps=8)
        forecast_hum = current['humidity'] + np.cumsum(hum_delta)
        forecast_hum = np.clip(forecast_hum, 10, 100)

        if 'precipprob' not in df.columns:
            trend = np.linspace(30, 70, len(df))
            noise = np.random.normal(0, 10, len(df))
            df['precipprob'] = np.clip(trend + noise, 0, 100)

        try:
            precip_model = SARIMAX(df['precipprob'].last('7D').diff().dropna(), order=(2, 0, 2), seasonal_order=(1, 0, 1, 24)).fit(disp=False)
            precip_delta = precip_model.forecast(steps=8)
        except:
            precip_delta = np.random.normal(0, 3, 8)

        forecast_precip = current['precipprob'] + np.cumsum(precip_delta)
        forecast_precip = np.clip(forecast_precip, 0, 100)

        # Time labels
        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        future_times = [(now + timedelta(hours=i+1)).strftime("%H:00") for i in range(8)]

        # Context setup
        temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8 = forecast_temp
        humidity1, humidity2, humidity3, humidity4, humidity5, humidity6, humidity7, humidity8 = forecast_hum
        precip1, precip2, precip3, precip4, precip5, precip6, precip7, precip8 = forecast_precip
        time1, time2, time3, time4, time5, time6, time7, time8 = future_times

        context = {
            'location': city,
            'tempmin': current['temp_min'],
            'tempmax': current['temp_max'],
            'temp': current['temp'],
            'humidity': current['humidity'],
            'windgust': current['windgust'],
            'windspeed': current['windspeed'],
            'winddir': current['winddir'],
            'sealevelpressure': 1010,
            'feels_like': current['feels_like'],
            'description': current['description'],
            'country': current['country'],
            'visibility': current['visibility'],
            'time': datetime.now(),
            'date': datetime.now().strftime("%B %d, %Y"),

            'time1': time1, 'time2': time2, 'time3': time3,
            'time4': time4, 'time5': time5, 'time6': time6,
            'time7': time7, 'time8': time8,

            'temp1': f"{round(temp1, 1)}", 'temp2': f"{round(temp2, 1)}", 'temp3': f"{round(temp3, 1)}",
            'temp4': f"{round(temp4, 1)}", 'temp5': f"{round(temp5, 1)}", 'temp6': f"{round(temp6, 1)}",
            'temp7': f"{round(temp7, 1)}", 'temp8': f"{round(temp8, 1)}",

            'humidity1': f"{round(humidity1, 1)}", 'humidity2': f"{round(humidity2, 1)}",
            'humidity3': f"{round(humidity3, 1)}", 'humidity4': f"{round(humidity4, 1)}",
            'humidity5': f"{round(humidity5, 1)}", 'humidity6': f"{round(humidity6, 1)}",
            'humidity7': f"{round(humidity7, 1)}", 'humidity8': f"{round(humidity8, 1)}",

            'precip1': f"{round(precip1, 1)}", 'precip2': f"{round(precip2, 1)}",
            'precip3': f"{round(precip3, 1)}", 'precip4': f"{round(precip4, 1)}",
            'precip5': f"{round(precip5, 1)}", 'precip6': f"{round(precip6, 1)}",
            'precip7': f"{round(precip7, 1)}", 'precip8': f"{round(precip8, 1)}",
        }

        return render(request, 'forecast/sarima.html', context)
    return render(request, 'forecast/sarima.html')

#...............VAR MODEL...............#
def var_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')

        # Load data
        df = pd.read_csv('C:\\Users\\shreeyaa\\.vscode\\Newfolder\\weatherpersonalised\\asia_01_20.csv')  # ✅ Update with your file path
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.asfreq('H').fillna(method='ffill')

        # Fetch live weather
        API_KEY = '5cb5519e0f137832e21b7a77ca27b51f'
        url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            return render(request, 'weather.html', {'error': f"City not found: {city}"})

        current = {
            'city': city.title(),
            'country': data['sys']['country'],
            'temp': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'temp_min': data['main']['temp_min'],
            'temp_max': data['main']['temp_max'],
            'humidity': data['main']['humidity'],
            'rain_last_hour': data.get('rain', {}).get('1h', 0),
            'precipprob': np.random.uniform(10, 90),
            'cloudcover': data['clouds']['all'],
            'windgust': data['wind'].get('gust', 0),
            'windspeed': data['wind']['speed'],
            'winddir': data['wind'].get('deg', 0),
            'visibility': data.get('visibility', 10000),
            'description': data['weather'][0]['description']
        }

        # Fill in missing columns
        for col in ['temp', 'humidity', 'precipprob']:
            if col not in df.columns:
                if col == 'precipprob':
                    df[col] = np.clip(np.linspace(30, 70, len(df)) + np.random.normal(0, 10, len(df)), 0, 100)
                else:
                    df[col] = np.random.normal(25, 5, len(df))

        var_data = df[['temp', 'humidity', 'precipprob']].last('7D').dropna()
        model = VAR(var_data)
        results = model.fit(maxlags=24)
        lag_order = results.k_ar
        last_obs = var_data.values[-lag_order:]
        forecast = results.forecast(last_obs, steps=8)
        forecast_df = pd.DataFrame(forecast, columns=['temp', 'humidity', 'precipprob'])
        forecast_df.index = pd.date_range(start=datetime.now(pytz.timezone('Asia/Kolkata')) + timedelta(hours=1), periods=8, freq='H')

        for col in ['temp', 'humidity', 'precipprob']:
            forecast_df[col] = current[col] + (forecast_df[col] - forecast_df[col].iloc[0])

        forecast_df['temp'] = np.clip(forecast_df['temp'], 10, 45)
        forecast_df['humidity'] = np.clip(forecast_df['humidity'], 10, 100)
        forecast_df['precipprob'] = np.clip(forecast_df['precipprob'], 0, 100)

        forecast_temp = forecast_df['temp'].values
        forecast_hum = forecast_df['humidity'].values
        forecast_precip = forecast_df['precipprob'].values
        future_times = [t.strftime("%H:%M") for t in forecast_df.index]

        temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8 = forecast_temp
        humidity1, humidity2, humidity3, humidity4, humidity5, humidity6, humidity7, humidity8 = forecast_hum
        precip1, precip2, precip3, precip4, precip5, precip6, precip7, precip8 = forecast_precip
        time1, time2, time3, time4, time5, time6, time7, time8 = future_times

        context = {
            'location': city,
            'tempmin': current['temp_min'],
            'tempmax': current['temp_max'],
            'temp': current['temp'],
            'humidity': current['humidity'],
            'windgust': current['windgust'],
            'windspeed': current['windspeed'],
            'winddir': current['winddir'],
            'sealevelpressure': 1010,
            'feels_like': current['feels_like'],
            'description': current['description'],
            'country': current['country'],
            'visibility': current['visibility'],
            'time': datetime.now(),
            'date': datetime.now().strftime("%B %d, %Y"),

            'time1': time1, 'time2': time2, 'time3': time3,
            'time4': time4, 'time5': time5, 'time6': time6,
            'time7': time7, 'time8': time8,

            'temp1': f"{round(temp1, 1)}", 'temp2': f"{round(temp2, 1)}", 'temp3': f"{round(temp3, 1)}",
            'temp4': f"{round(temp4, 1)}", 'temp5': f"{round(temp5, 1)}", 'temp6': f"{round(temp6, 1)}",
            'temp7': f"{round(temp7, 1)}", 'temp8': f"{round(temp8, 1)}",

            'humidity1': f"{round(humidity1, 1)}", 'humidity2': f"{round(humidity2, 1)}",
            'humidity3': f"{round(humidity3, 1)}", 'humidity4': f"{round(humidity4, 1)}",
            'humidity5': f"{round(humidity5, 1)}", 'humidity6': f"{round(humidity6, 1)}",
            'humidity7': f"{round(humidity7, 1)}", 'humidity8': f"{round(humidity8, 1)}",

            'precip1': f"{round(precip1, 1)}", 'precip2': f"{round(precip2, 1)}",
            'precip3': f"{round(precip3, 1)}", 'precip4': f"{round(precip4, 1)}",
            'precip5': f"{round(precip5, 1)}", 'precip6': f"{round(precip6, 1)}",
            'precip7': f"{round(precip7, 1)}", 'precip8': f"{round(precip8, 1)}",
        }

        return render(request, 'forecast/var.html', context)
    return render(request, 'forecast/var.html')

#...............VARIMA MODEL.............#
def varima_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')

        # Load historical data
        df = pd.read_csv('C:\\Users\\shreeyaa\\.vscode\\Newfolder\\weatherpersonalised\\asia_01_20.csv')  # ✅ Update to correct path
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.asfreq('H').fillna(method='ffill')

        # Simulate precipprob if missing
        if 'precipprob' not in df.columns:
            np.random.seed(42)
            df['precipprob'] = np.random.uniform(20, 80, size=len(df))

        # Fetch live weather
        API_KEY = '5cb5519e0f137832e21b7a77ca27b51f'
        url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            return render(request, 'weather.html', {'error': f"City not found: {city}"})

        # Live weather current
        current = {
            'city': city.title(),
            'country': data['sys']['country'],
            'temp': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'temp_min': data['main']['temp_min'],
            'temp_max': data['main']['temp_max'],
            'humidity': data['main']['humidity'],
            'rain_last_hour': data.get('rain', {}).get('1h', 0),
            'precipprob': np.random.uniform(10, 90),
            'cloudcover': data['clouds']['all'],
            'windgust': data['wind'].get('gust', 0),
            'windspeed': data['wind']['speed'],
            'winddir': data['wind'].get('deg', 0),
            'visibility': data.get('visibility', 10000),
            'description': data['weather'][0]['description']
        }

        # Forecast using VARIMA
        df_recent = df[['temp', 'humidity', 'precipprob']].last('30D')
        delta_df = df_recent.diff().dropna()
        model = VARMAX(delta_df, order=(2, 0))
        model_fit = model.fit(disp=False)
        delta_forecast = model_fit.forecast(steps=8)

        forecast = pd.DataFrame()
        forecast['temp'] = current['temp'] + np.cumsum(delta_forecast['temp'].values)
        forecast['humidity'] = current['humidity'] + np.cumsum(delta_forecast['humidity'].values)
        forecast['precipprob'] = current['precipprob'] + np.cumsum(delta_forecast['precipprob'].values)

        forecast['temp'] = np.clip(forecast['temp'], 5, 45)
        forecast['humidity'] = np.clip(forecast['humidity'], 10, 100)
        forecast['precipprob'] = np.clip(forecast['precipprob'], 0, 100)

        future_times = [(datetime.now(pytz.timezone('Asia/Kolkata')) + timedelta(hours=i+1)).strftime('%H:%M') for i in range(8)]

        # Unpack forecast values
        temp = forecast['temp'].values
        humidity = forecast['humidity'].values
        precip = forecast['precipprob'].values

        context = {
            'location': city,
            'tempmin': current['temp_min'],
            'tempmax': current['temp_max'],
            'temp': current['temp'],
            'humidity': current['humidity'],
            'windgust': current['windgust'],
            'windspeed': current['windspeed'],
            'winddir': current['winddir'],
            'sealevelpressure': 1010,
            'feels_like': current['feels_like'],
            'description': current['description'],
            'country': current['country'],
            'visibility': current['visibility'],
            'time': datetime.now(),
            'date': datetime.now().strftime("%B %d, %Y"),

            'time1': future_times[0], 'time2': future_times[1], 'time3': future_times[2], 'time4': future_times[3],
            'time5': future_times[4], 'time6': future_times[5], 'time7': future_times[6], 'time8': future_times[7],

            'temp1': f"{round(temp[0], 1)}", 'temp2': f"{round(temp[1], 1)}", 'temp3': f"{round(temp[2], 1)}",
            'temp4': f"{round(temp[3], 1)}", 'temp5': f"{round(temp[4], 1)}", 'temp6': f"{round(temp[5], 1)}",
            'temp7': f"{round(temp[6], 1)}", 'temp8': f"{round(temp[7], 1)}",

            'humidity1': f"{round(humidity[0], 1)}", 'humidity2': f"{round(humidity[1], 1)}",
            'humidity3': f"{round(humidity[2], 1)}", 'humidity4': f"{round(humidity[3], 1)}",
            'humidity5': f"{round(humidity[4], 1)}", 'humidity6': f"{round(humidity[5], 1)}",
            'humidity7': f"{round(humidity[6], 1)}", 'humidity8': f"{round(humidity[7], 1)}",

            'precip1': f"{round(precip[0], 1)}", 'precip2': f"{round(precip[1], 1)}",
            'precip3': f"{round(precip[2], 1)}", 'precip4': f"{round(precip[3], 1)}",
            'precip5': f"{round(precip[4], 1)}", 'precip6': f"{round(precip[5], 1)}",
            'precip7': f"{round(precip[6], 1)}", 'precip8': f"{round(precip[7], 1)}",
        }

        return render(request, 'forecast/varima.html', context)
    return render(request, 'forecast/varima.html')

#.............COMPARE MODEL..........#
from django.shortcuts import render
from .views import arima_view, sarima_view, var_view, varima_view, weather_view  # import your model views

def compare_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        param = request.POST.get('parameter')

        # Call all 5 model forecast functions
        temp_forecast_arima, hum_forecast_arima, precip_forecast_arima, future_times, city_name_arima = arima_view(city)
        temp_forecast_sarima, hum_forecast_sarima, precip_forecast_sarima, future_times_sarima, city_name_sarima = sarima_view(city)
        var_temp, var_humidity, var_precip, future_times_var, city_name_var = var_view(city)
        temp_varima, hum_varima, precip_varima, future_times_varima, city_name_varima = varima_view(city)
        linear_forecast_temp, linear_forecast_humidity, linear_forecast_precip, future_times_, city_name_lr = weather_view(city)

        # Determine which parameter the user selected
        if param == 'temp':
            title = f"🌡️ Temperature Forecast Comparison – {city.title()}"
            y1 = temp_forecast_arima
            y2 = temp_forecast_sarima
            y3 = var_temp
            y4 = temp_varima
            y5 = linear_forecast_temp
        elif param == 'humidity':
            title = f"💧 Humidity Forecast Comparison – {city.title()}"
            y1 = hum_forecast_arima
            y2 = hum_forecast_sarima
            y3 = var_humidity
            y4 = hum_varima
            y5 = linear_forecast_humidity
        elif param == 'precipprob':
            title = f"🌧️ Precipitation Probability Comparison – {city.title()}"
            y1 = precip_forecast_arima
            y2 = precip_forecast_sarima
            y3 = var_precip
            y4 = precip_varima
            y5 = linear_forecast_precip
        else:
            return render(request, 'forecast/compare.html', {
                'error': 'Invalid parameter selected.'
            })

        # Render compare.html with Chart.js data
        return render(request, 'forecast/compare.html', {
            'city': city.title(),
            'param': param,
            'title': title,
            'times': future_times,
            'y1': y1,
            'y2': y2,
            'y3': y3,
            'y4': y4,
            'y5': y5,
        })

    # For GET requests
    return render(request, 'forecast/compare.html')
