import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 종목 이름 정의
ticker = 'NVDA'

# 야후 파이낸스에서 데이터 다운로드
end_date = datetime.now()
start_date = end_date - timedelta(days=3650)
data = yf.download(ticker, start=start_date, end=end_date)

closing_prices = data['Close']

# MACD 계산 함수


def calculate_macd(closing_prices):
    # 26일 지수 이동 평균 (단기 이동 평균)
    short_ema = closing_prices.ewm(span=26, adjust=False).mean()
    # 12일 지수 이동 평균(장기 이동평균)
    long_ema = closing_prices.ewm(span=12, adjust=False).mean()
    # DIF : 단기 이동평균과 장기 이동평균 간의 차이
    dif = short_ema - long_ema
    # Signal Line : DIF의 9일 이동평균
    signal_line = dif.ewm(span=9, adjust=False).mean()
    # histogram : DIF와 Signal Line 간의 차이
    histogram = dif - signal_line
    return dif, signal_line, histogram

# MACD 계산
dif, signal_line, histogram = calculate_macd(closing_prices)
macd_data = pd.DataFrame({'DIF': dif, 'Signal Line': signal_line, 'Histogram': histogram})

# Result, Prediction 계산
macd_data['Result'] = np.where(macd_data['DIF'] > macd_data['Signal Line'], 1, 0)
macd_data['Result'] = macd_data['Result'].rolling(window=20, min_periods=1).mean()
macd_data['Prediction'] = macd_data['Result'].shift(-1)
macd_data.dropna(inplace=True)

# Trend 계산
macd_data['Trend'] = macd_data['Prediction'].diff()
macd_data['Trend'] = (macd_data['Trend'] + 1) / 2

# 전체 예측을 기반으로 추세를 계산하여 출력
future_trend = macd_data['Prediction'].diff().sum()
MACD_result = (future_trend + 1) / 2
print("MACD result:", 100*MACD_result)


# STCK 계산 함수 (20일 이동평균 사용)


def calculate_stck(closing_prices, window=20):
    # 최저 가격과 최고 가격을 계산
    lowest_low = closing_prices.rolling(window=window).min()
    highest_high = closing_prices.rolling(window=window).max()
    stck = 100 * (closing_prices - lowest_low) / (highest_high - lowest_low)
    stck_ma = stck.rolling(window=20).mean()
    return stck_ma

# STCK 계산
prediction = calculate_stck(closing_prices)

# NaN 값을 가진 행 제거
prediction.dropna(inplace=True)

# STCK 값의 평균을 사용하여 최종 예측 계산
STCK_result = prediction.mean()

# "미래 추세"만 출력
print("STCK_result:", STCK_result)

# Bollinger Bands 계산 함수
def calculate_bollinger_bands(closing_prices, window=20, num_std=2):
    rolling_mean = closing_prices.rolling(window=window).mean()
    rolling_std = closing_prices.rolling(window=window).std()

    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std

    return upper_band, lower_band


# Bollinger Bands 계산


upper_band, lower_band = calculate_bollinger_bands(closing_prices)
bollinger_bands_data = pd.DataFrame({'Upper Band': upper_band, 'Lower Band': lower_band})

# NaN 값을 가지는 행 제거
bollinger_bands_data.dropna(inplace=True)

# 'Result' 계산
result = np.where(closing_prices > upper_band, (closing_prices - upper_band) / (upper_band - lower_band), np.where(closing_prices < lower_band, (closing_prices - lower_band) / (lower_band - upper_band), np.nan))
bollinger_bands_data['Result'] = pd.Series(result, index=closing_prices.index)

# 'Result' 값을 NaN 값을 직전의 유효한 값으로 채우기
bollinger_bands_data['Result'] = bollinger_bands_data['Result'].ffill()

# 'Prediction' 계산
bollinger_bands_data['Prediction'] = bollinger_bands_data['Result'].shift(-1)

# NaN 값을 가지는 행 제거
bollinger_bands_data.dropna(inplace=True)

# 'Trend' 계산
bollinger_bands_data['Trend'] = bollinger_bands_data['Prediction'].diff()
bollinger_bands_data['Trend'] = (bollinger_bands_data['Trend'] + 1) / 2

# 'Prediction' 및 'Trend' 열 정규화
min_pred = bollinger_bands_data['Prediction'].min()
max_pred = bollinger_bands_data['Prediction'].max()
bollinger_bands_data['Prediction'] = (bollinger_bands_data['Prediction'] - min_pred) / (max_pred - min_pred)

min_trend = bollinger_bands_data['Trend'].min()
max_trend = bollinger_bands_data['Trend'].max()
bollinger_bands_data['Trend'] = (bollinger_bands_data['Trend'] - min_trend) / (max_trend - min_trend)

future_trend = bollinger_bands_data['Prediction'].iloc[-1] - bollinger_bands_data['Prediction'].iloc[0]
Bollinger_result = (future_trend + 1) / 2
print("Bollinger_result:", 100*Bollinger_result)

# RSI를 계산하는 함수


def calculate_rsi(closing_prices, window=20):
    # 전일 대비 가격 변화를 계산합니다.
    price_changes = closing_prices.diff()

    # 상승 가격과 하락 가격을 계산합니다.
    up_changes = price_changes.where(price_changes > 0, 0)
    down_changes = -price_changes.where(price_changes < 0, 0)

    # 지정된 기간 동안의 평균 상승 및 하락 가격을 계산합니다.
    avg_up_changes = up_changes.rolling(window=window, min_periods=1).mean()
    avg_down_changes = down_changes.rolling(window=window, min_periods=1).mean()

    # RS (상대 강도)를 계산합니다.
    rs = avg_up_changes / avg_down_changes

    # 공식을 사용하여 RSI를 계산합니다.
    rsi = 100 - (100 / (1 + rs))

    return rsi

# 동적 기간으로 RSI를 계산합니다.
rsi_window = len(closing_prices) // 10
prediction = calculate_rsi(closing_prices, window=rsi_window)

# 최신 RSI 값을 표시합니다 (미래 추세).
RSI_result = prediction.iloc[-1]
print("RSI_result:", RSI_result)


#VR


# 분산 비율을 계산하는 함수
def calculate_variance_ratio(closing_prices, short_window=20, long_window=120):
    returns = closing_prices.pct_change()

    # 계산에 필요한 충분한 수의 수익률 데이터가 있는지 확인합니다.
    if len(returns) < long_window:
        return np.nan

    # 단기 및 장기 분산을 계산합니다.
    short_variance = returns[-short_window:].var()
    long_variance = returns[-long_window:].var()

    # 0으로 나누거나 NaN 결과를 처리합니다.
    if short_variance == 0 or np.isnan(short_variance) or np.isnan(long_variance):
        return np.nan

    # 분산 비율을 계산합니다.
    variance_ratio = long_variance / short_variance

    return variance_ratio


# 각 시간 기간에 대한 분산 비율을 계산합니다.
window_size = 120
vr_values = []
for i in range(window_size, len(closing_prices) + 1):
    vr = calculate_variance_ratio(closing_prices[i - window_size:i])
    vr_values.append(vr)

# 분산 비율 값을 0에서 100 사이로 정규화합니다.
min_vr = np.nanmin(vr_values)
max_vr = np.nanmax(vr_values)
normalized_vr = 100 * (vr_values - min_vr) / (max_vr - min_vr)

# 정규화된 분산 비율 값과 최종 예측을 계산합니다.
VR_result = np.nanmean(normalized_vr)
print("VR_result:", VR_result)


# WR


# 20일 이동 평균을 사용하여 Williams %R을 계산하는 함수
def calculate_williams_r(closing_prices, period=14):
    high_prices = data['High']
    low_prices = data['Low']

    # 종가의 20일 이동평균을 계산합니다.
    moving_average = closing_prices.rolling(window=20).mean()

    # 주어진 기간 동안의 가장 높은 고가와 가장 낮은 저가를 계산합니다.
    highest_high = high_prices.rolling(window=period).max()
    lowest_low = low_prices.rolling(window=period).min()

    # 20일 이동평균을 사용하여 Williams %R을 계산합니다.
    williams_r = (highest_high - moving_average) / (highest_high - lowest_low) * -100

    # 값을 0에서 100 사이로 정규화합니다.
    normalized_williams_r = 100 * (williams_r + 100) / 100

    return normalized_williams_r

# Williams %R 값을 계산합니다.
williams_r_values = calculate_williams_r(closing_prices)

# 정규화된 Williams %R 값을 사용하여 최종 예측을 계산합니다.
WR_result = np.nanmean(williams_r_values)
print("WR_result :", WR_result )


# MOK 


# MOK (모멘텀) 지표를 20일 이동평균을 사용하여 계산하는 함수
def calculate_mok(closing_prices, period=14, ma_period=20):
    returns = closing_prices.pct_change()

    # 종가의 20일 이동평균을 계산합니다.
    moving_average = closing_prices.rolling(window=ma_period).mean()

    # 모멘텀을 현재 가격과 n 기간 전 가격의 차이로 계산합니다.
    momentum = closing_prices.diff(period)

    # 모멘텀 값을 0에서 100 사이로 정규화합니다.
    normalized_momentum = 100 * (momentum - np.min(momentum)) / (np.max(momentum) - np.min(momentum))

    return normalized_momentum

# MOK (모멘텀) 값을 계산합니다.
mok_values = calculate_mok(closing_prices)

# 정규화된 MOK (모멘텀) 값을 사용하여 최종 예측을 계산합니다.
MOK_result = np.nanmean(mok_values)
print("MOK_result:", MOK_result)