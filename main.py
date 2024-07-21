from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import os
import pandas as pd

from LinearRegressionModel import LinearRegressionModel

if os.path.isfile("./crude_oil_data.csv"):
    df = pd.read_csv("./crude_oil_data.csv")

    linear_model = LinearRegressionModel(df)

    linear_model.show_diagram()
    linear_model.train_and_predict()

else:
    URL = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=pet&s=f000000__3&f=m"

    chrome_options = webdriver.ChromeOptions()

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    driver.get(URL)

    table = driver.find_element(By.ID, 'tmpFloatTitleTableId0')

    thead = table.find_element(By.TAG_NAME, 'thead')
    header_cells = thead.find_element(By.TAG_NAME, 'tr').find_elements(By.TAG_NAME, 'th')
    header_data = [cell.text for cell in header_cells]

    tbody = table.find_element(By.TAG_NAME, 'tbody')
    rows = tbody.find_elements(By.TAG_NAME, 'tr')

    table_data = []

    for row in rows:
        cells = row.find_elements(By.TAG_NAME, 'td')
        row_data = [cell.text for cell in cells]
        table_data.append(row_data)

    crude_oil_df = pd.DataFrame(table_data, columns=header_data)

    crude_oil_df.dropna(subset=header_data[1:], inplace=True)

    crude_oil_df = pd.melt(crude_oil_df, id_vars=['Year'], var_name='Month', value_name='Value')

    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    crude_oil_df['Month'] = pd.Categorical(crude_oil_df['Month'], categories=month_order, ordered=True)
    crude_oil_df['Month'] = pd.to_datetime(crude_oil_df['Month'], format='%b').dt.month

    crude_oil_df.sort_values(by=['Year', 'Month'], inplace=True)

    crude_oil_df = crude_oil_df[crude_oil_df['Value'] != '']

    crude_oil_df.to_csv('crude_oil_data.csv', index=False)

    linear_model = LinearRegressionModel(crude_oil_df)

    linear_model.show_diagram()
    linear_model.train_and_predict()

    driver.quit()
