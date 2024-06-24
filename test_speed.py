import pandas as pd
from sklearn.model_selection import train_test_split
from model import CustomLogReg

import time



df = pd.read_parquet('data/casia_webface/data.parquet')

df_pure = pd.read_parquet('data/casia_webface/data_pure.parquet')

df = df[~df['name'].isin(df_pure['name'].map(lambda x: x[:6]).unique())].copy()

df = pd.concat([df, df_pure], axis=0)

test_names = pd.DataFrame({'name': df['name'].unique()[:375], 'test': True})

df = pd.merge(test_names, df, on='name', how='right').fillna(False)

df_test = df[df['test']].copy()

df = df[~df['test']].copy()


start_reading_data = time.time()


features = [item for item in df.columns if 'emb' in item]
keys = [item for item in df.columns if 'key' in item]
X_train, X_valid, Y_train, Y_valid = train_test_split(df, df[keys], test_size=0.3, random_state=0)
Y_train, Y_valid = Y_train.to_numpy(), Y_valid.to_numpy()

X_test, Y_test = df_test, df_test[keys].to_numpy().copy()



log_reg = CustomLogReg(lambda_l2=1e-8)

start_fit = time.time()

log_reg.fit(X_train[features], Y_train)

finish_fit = time.time()

print('read + fit:', round(finish_fit - start_reading_data, 4), 'sec')

print('fit:', round(finish_fit - start_fit, 4), 'sec')


start_predict = time.time()

log_reg.predict(X_test.iloc[0:1][features])

print('finish_predict:', round(time.time() - start_predict, 4), 'sec')
