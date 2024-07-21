import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from datetime import datetime
from tqdm import tqdm
import time


class LinearRegressionModel:
    def __init__(self, df):
        self.temp_df = df.copy()
        self.temp_df['Date'] = pd.to_datetime(self.temp_df[['Year', 'Month']].assign(DAY=1))
        self.temp_df['Time'] = (self.temp_df['Date'].dt.year - self.temp_df['Date'].dt.year.min()) * 12 + self.temp_df['Date'].dt.month

        self.X = self.temp_df[['Time']]
        self.y = self.temp_df['Value']

    def time_to_date(self, time, start_year):
        year = start_year + (time - 1) // 12
        month = (time - 1) % 12 + 1

        return datetime(year, month, 1)

    def show_diagram(self):
        temp_df = self.temp_df.copy()

        start_year = temp_df['Date'].dt.year.min()
        temp_df['Converted_Date'] = temp_df['Time'].apply(lambda x: self.time_to_date(x, start_year))
        temp_df['Date_Label'] = temp_df['Converted_Date'].dt.strftime('%m-%Y')

        plt.figure(figsize=(12, 6))
        plt.plot(temp_df['Converted_Date'], temp_df['Value'], marker='o')
        plt.xlabel('Date')
        plt.ylabel('Crude Oil Price')
        plt.title('Crude Oil Prices Over Time')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def train_and_predict(self):
        # Train
        model = LinearRegressionWithProgress()
        model.fit(self.X, self.y)

        # Predict
        future_date = pd.to_datetime({'year': [2024], 'month': [5], 'day': [1]})
        future_time = (future_date.dt.year - self.temp_df['Date'].dt.year.min()) * 12 + future_date.dt.month

        predicted_value = model.predict(future_time.values.reshape(-1, 1))
        print(f"Predicted value for May 2024: {predicted_value[0]:.2f}")


class LinearRegressionWithProgress(LinearRegression):
    def fit(self, X, y):
        n_iter = len(X)
        for i in tqdm(range(n_iter), desc="Fitting Progress"):
            super().fit(X.iloc[:i+1], y.iloc[:i+1])
            time.sleep(0.01)
