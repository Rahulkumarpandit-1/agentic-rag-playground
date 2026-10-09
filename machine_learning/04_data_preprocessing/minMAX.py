from sklearn.preprocessing import MinMaxScaler
import pandas   as pd
data = pd.DataFrame({
    "age": [20, 25, 30, 35, 40],
    "salary": [20000, 40000, 60000, 80000, 100000]
})

scaler = MinMaxScaler()

scaled = scaler.fit_transform(data)

print(scaled)
