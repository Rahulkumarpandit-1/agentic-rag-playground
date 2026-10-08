import pandas as pd
from sklearn.preprocessing import OneHotEncoder     


df = pd.DataFrame({
    "age": [21, 25, 30, 22],
    "city": ["Delhi", "Mumbai", "Guwahati", "Delhi"]
})

encoder=OneHotEncoder(sparse_output=False)

encoded=encoder.fit_transform(df[["city"]])

encoded_df=pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["city"])
     
)
print(encoded_df)