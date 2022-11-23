from pathlib import Path
import pandas as pd
import os

date = 23111960
time = 124500
i = 0

path = Path('data') / 'default of credit card clients.xls'
new_path = Path('Training_Batch_Files')

data = pd.read_excel(path)
rows, columns = data.shape

for j in range(0, (rows//1000)+1):
    new = data.iloc[i:i + 1000]
    name = "creditCardFraud_" + str(date) + '_' + str(time) + ".csv"
    destination = os.path.join(new_path, name)
    new.to_csv(destination, index=None, header=True)
    i += 1
    date += 1
    time += 1
