import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {
    'value': [100, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
}
df = pd.DataFrame(data)
bins = [1, 4000, 8000, 12000, 16000, 20000]
labels = ['1-4000', '4000-8000', '8000-12000', '12000-16000', '16000-20000']
df['binned'] = pd.cut(df['value'], bins=bins, labels=labels, include_lowest=True)
print("数据封箱结果：")
print(df)
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(df[['binned']])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.categories_[0])
df = pd.concat([df, encoded_df], axis=1)

print("\n封箱和 One-Hot 编码后的结果：")
print(df)
