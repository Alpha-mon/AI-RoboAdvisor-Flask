# 전체 코드

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 야후 파이낸스 주식 정보 가져오기
# 주식 종목 지정 필수

ticker = 'NVDA'
end_date = datetime.now()
start_date = end_date - timedelta(days=3650)
data = yf.download(ticker, start=start_date, end=end_date)

closing_prices = data['Close']
volume = data['Volume']
moving_average = closing_prices.rolling(window=120).mean()

# 6개 알고리즘 (Bollinger, MACD, MOK, RSI, STCK, WR) 전체 코드


# Bollinger

def calculate_bollinger_bands(closing_prices, window=20, num_std=2):
    rolling_mean = closing_prices.rolling(window=window).mean()
    rolling_std = closing_prices.rolling(window=window).std()

    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    return upper_band, lower_band

upper_band, lower_band = calculate_bollinger_bands(closing_prices)
bollinger_bands_data = pd.DataFrame({'Upper Band': upper_band, 'Lower Band': lower_band})

bollinger_bands_data.dropna(inplace=True)
result = np.where(closing_prices > upper_band, (closing_prices - upper_band) / (upper_band - lower_band), np.where(closing_prices < lower_band, (closing_prices - lower_band) / (lower_band - upper_band), np.nan))
bollinger_bands_data['Result'] = pd.Series(result, index=closing_prices.index)

bollinger_bands_data['Result'].fillna(method='ffill', inplace=True)

bollinger_bands_data['Prediction'] = bollinger_bands_data['Result'].shift(-1)
bollinger_bands_data.dropna(inplace=True)
bollinger_bands_data['Trend'] = bollinger_bands_data['Prediction'].diff()
bollinger_bands_data['Trend'] = (bollinger_bands_data['Trend'] + 1) / 2

min_pred = bollinger_bands_data['Prediction'].min()
max_pred = bollinger_bands_data['Prediction'].max()
bollinger_bands_data['Prediction'] = (bollinger_bands_data['Prediction'] - min_pred) / (max_pred - min_pred)

min_trend = bollinger_bands_data['Trend'].min()
max_trend = bollinger_bands_data['Trend'].max()
bollinger_bands_data['Trend'] = (bollinger_bands_data['Trend'] - min_trend) / (max_trend - min_trend)*100

pd.set_option('display.float_format', '{:.4f}'.format)


# MACD

def calculate_macd(closing_prices):

    short_ema = closing_prices.ewm(span=26, adjust=False).mean()
    long_ema = closing_prices.ewm(span=12, adjust=False).mean()

    dif = short_ema - long_ema
    signal_line = dif.ewm(span=9, adjust=False).mean()
    histogram = dif - signal_line

    return dif, signal_line, histogram

dif, signal_line, histogram = calculate_macd(closing_prices)
macd_data = pd.DataFrame({'DIF': dif, 'Signal Line': signal_line, 'Histogram': histogram})
macd_data['Result'] = np.where(macd_data['DIF'] > macd_data['Signal Line'], 1, 0)

macd_data['Result'] = (macd_data['Result'].rolling(window=20, min_periods=1).mean())*100
macd_data['Prediction'] = macd_data['Result'].shift(-1)
macd_data.dropna(inplace=True)

# MOK

def calculate_mok(closing_prices, period=14, ma_period=20):
    returns = closing_prices.pct_change()

    moving_average = closing_prices.rolling(window=ma_period).mean()
    momentum = closing_prices.diff(period)
    normalized_momentum = 100 * (momentum - np.min(momentum)) / (np.max(momentum) - np.min(momentum))

    return normalized_momentum

mok_values = calculate_mok(closing_prices)

# RSI

def calculate_rsi(closing_prices, window=20):
    price_changes = closing_prices.diff()
    up_changes = price_changes.where(price_changes > 0, 0)
    down_changes = -price_changes.where(price_changes < 0, 0)

    avg_up_changes = up_changes.rolling(window=window, min_periods=1).mean()
    avg_down_changes = down_changes.rolling(window=window, min_periods=1).mean()

    rs = avg_up_changes / avg_down_changes
    rsi = 100 - (100 / (1 + rs))

    return rsi

rsi_result = calculate_rsi(closing_prices, window=len(closing_prices)//10)  # 기간을 데이터의 1/10로 동적으로 설정
rsi_result.dropna(inplace=True)

# STCK

def calculate_stck(closing_prices, window=20):
    lowest_low = closing_prices.rolling(window=window).min()
    highest_high = closing_prices.rolling(window=window).max()

    stck = 100 * (closing_prices - lowest_low) / (highest_high - lowest_low)

    stck_ma = stck.rolling(window=20).mean()

    return stck_ma

stck_result = calculate_stck(closing_prices)
stck_result.dropna(inplace=True)

# WR

def calculate_williams_r(closing_prices, period=14):
    high_prices = data['High']
    low_prices = data['Low']
    moving_average = closing_prices.rolling(window=20).mean()

    highest_high = high_prices.rolling(window=period).max()
    lowest_low = low_prices.rolling(window=period).min()

    williams_r = (highest_high - moving_average) / (highest_high - lowest_low) * -100

    normalized_williams_r = 100 * (williams_r + 100) / 100

    return normalized_williams_r

wr_result = calculate_williams_r(closing_prices)

# 6개 지표들을 하나의 데이터 프레임에 합치기

final_df = pd.DataFrame({
    'Bollinger': bollinger_bands_data['Trend'],
    'MACD': macd_data['Prediction'],
    'MOK': mok_values,
    'RSI': rsi_result,
    'STCK': stck_result,
    'WR': wr_result
})

# NaN 값 제거
final_df.dropna(inplace=True)

# 최종 데이터 프레임 출력
print(final_df)

# LSTM 모델


final_df_values = final_df.values
data_values = data.values

# 입력 시퀀스에 대한 타임 스텝(T)을 정의합니다.

T = 10  # 원하는대로 조정할 수 있습니다.

# 입력 및 타겟을 위한 데이터 시퀀스 생성
final_df_sequences = []
data_sequences = []

for i in range(len(data_values) - T):
    final_df_sequences.append(final_df_values[i:i+T])
    data_sequences.append(closing_prices.iloc[i+T])

filtered_final_df_sequences = []
filtered_data_sequences = []

# 시퀀스 길이 맞추기
for i, seq in enumerate(final_df_sequences):
    if len(seq) == 10:
        filtered_final_df_sequences.append(seq)
        filtered_data_sequences.append(data_sequences[i])

final_df_sequences = filtered_final_df_sequences
data_sequences = filtered_data_sequences

# 시퀀스를 넘파이 배열로 변환
X = np.array(final_df_sequences)
y = np.array(data_sequences)


# 데이터를 훈련 및 테스트 세트로 분할
split_ratio = 0.8  # 분할 비율을 조정할 수 있습니다.
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# LSTM 모델 구축
model = Sequential()

# 첫 번째 LSTM 레이어 (시퀀스 출력을 반환하여 다음 레이어로 전달)
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(T, 6)))

# 두 번째 LSTM 레이어
model.add(LSTM(50, activation='relu', return_sequences=True))

# 세 번째 LSTM 레이어 (시퀀스 출력을 반환하지 않음)
model.add(LSTM(50, activation='relu'))

# 출력 레이어
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')


# 모델 훈련
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 테스트 데이터에서 모델 평가
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# 예측 수행
predictions = model.predict(X_test)
print(predictions)

# 예측 결과를 데이터 프레임으로 변환
predictions_df = pd.DataFrame(predictions, columns=["Predictions"])
print(predictions_df)

# LSTM 모델의 예측 값을 6개 지표 값과 함께 생성 모델 입력값으로 넣기 위해

# T 타임 스텝 만큼의 길이를 고려하여 LSTM 예측값 넣기
final_df["LSTM_Predictions"] = np.nan
final_df["LSTM_Predictions"].iloc[-len(predictions_df):] = predictions_df["Predictions"].values

# NaN 값 제거
final_df.dropna(inplace=True)

# 다시 시퀀스로 변환
final_df_values = final_df.values

final_df_sequences = []
for i in range(len(data_values) - T):
    final_df_sequences.append(final_df_values[i:i+T])

X = np.array(final_df_sequences)

# 생성 및 구분 모델

from keras.layers import Reshape, Flatten, LeakyReLU, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

# 생성 모델 G (주식 가격의 시퀀스를 생성)
def build_generator(input_shape=(T, 7)):  # 입력 값 : LSTM 모델 예측 + 6개 지표 값
    model = Sequential()

    # LSTM 레이어와 BatchNormalization, LeakyReLU 활성화 함수를 이용하여 시퀀스를 학습
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    # Flatten layer를 제거하고, T 길이의 시퀀스를 생성하도록 수정
    model.add(LSTM(128))
    model.add(Dense(T, activation='linear'))

    noise = tf.keras.layers.Input(shape=input_shape)
    generated_sequence = model(noise)

    return Model(noise, generated_sequence)

# 구분 모델 D (주가의 실제 시퀀스와 생성된 시퀀스를 구분)
def build_discriminator(input_shape=(T, 1)):
    model = Sequential()

    # LSTM 레이어와 LeakyReLU 활성화 함수를 사용하여 시퀀스를 처리
    model.add(LSTM(128, input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    sequence = tf.keras.layers.Input(shape=input_shape)
    validity = model(sequence)

    # 주어진 시퀀스가 실제인지 생성된 것인지에 대한 확률
    return Model(sequence, validity)

    # 학습 함수 (생성된 주식 가격 시퀀스와 실제 주식 가격 시퀀스를 사용하여 판별자를 학습)

def train_gan(generator, discriminator, combined, epochs, batch_size=32):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        sequences = X_train[idx]

        # 주식 예측값 생성
        predicted_stock = model.predict(sequences)

        # LSTM 예측값을 sequences에 추가
        predicted_stock_reshaped = predicted_stock.reshape(batch_size, 1, 1)  # (batch_size, 1, 1) 형태로 변환
        predicted_stock_expanded = np.repeat(predicted_stock_reshaped, T, axis=1)  # predicted_stock_reshaped를 T 타임 스텝만큼 확장
        X_train_combined = np.concatenate([sequences, predicted_stock_expanded], axis=2)

        generated_stock = generator.predict(X_train_combined)
        generated_stock_reshaped = generated_stock.reshape(batch_size, T, 1)

        # 판별자 학습
        d_loss_real = discriminator.train_on_batch(predicted_stock_expanded, valid)
        d_loss_fake = discriminator.train_on_batch(generated_stock_reshaped, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 생성자 학습
        g_loss = combined.train_on_batch(X_train_combined, valid)

        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        # 판별자 및 생성자 모델 초기화, 생성자와 판별자를 결합하여

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

generator = build_generator()
z = tf.keras.layers.Input(shape=(T, 7))
generated_sequence = generator(z)

discriminator.trainable = False
validity = discriminator(generated_sequence)

combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

#학습 시작
train_gan(generator, discriminator, combined, epochs=10, batch_size=32)

# 학습 후 최종 예측을 수행하는 함수
def final_predictions(generator, test_data, batch_size=32):
    predicted_stock_lstm = model.predict(test_data)

    # LSTM 예측값을 test_data에 추가
    predicted_stock_expanded = np.repeat(predicted_stock_lstm[:, np.newaxis], T, axis=1)  # (batch_size, T)로 형태 변경
    test_data_combined = np.concatenate([test_data, predicted_stock_expanded], axis=2)

    # G 모델을 사용하여 합성 주가를 생성
    generated_stock = generator.predict(test_data_combined)

    # LSTM 예측과 G 모델의 예측을 평균
    final_predicted_stock = (predicted_stock_lstm + generated_stock.mean(axis=1)) / 2.0

    return final_predicted_stock

# 학습 후 예측
after_gan_predictions = final_predictions(generator, X_test, batch_size=32)

# 전체 예측 출력
print(after_gan_predictions)

# 최종 예측 출력 (평균)
average_prediction = np.mean(after_gan_predictions)
print(average_prediction)
