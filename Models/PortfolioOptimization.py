import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 사용자로부터 보유 주식 입력 받기
stock_list = input("보유하고 있는 주식의 티커 목록을 쉼표로 구분하여 입력하세요 (예: AAPL,GOOGL,MSFT): ").split(",")
stock_list = [stock.strip() for stock in stock_list]

def fetch_data(tickers):
    data = {}
    for ticker in tickers:
        stock_data = yf.Ticker(ticker)
        data[ticker] = stock_data.history(period="1y")
    return data

stock_data = fetch_data(stock_list)

# 수익률 계산
returns = {stock: data['Close'].pct_change().dropna().values for stock, data in stock_data.items()}

look_back = 5
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

models = {}
for stock, daily_returns in returns.items():
    scaler = MinMaxScaler(feature_range=(0, 1))
    daily_returns = scaler.fit_transform(daily_returns.reshape(-1, 1))
    X, Y = create_dataset(daily_returns, look_back)

    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=1, verbose=2)
    models[stock] = model

predicted_volatilities = {}
for stock, model in models.items():
    last_sequence = returns[stock][-look_back:].reshape(-1, 1)
    scaled_sequence = scaler.transform(last_sequence)
    predicted_return = model.predict(scaled_sequence.reshape(1, look_back, 1))
    predicted_volatility = scaler.inverse_transform(predicted_return)[0][0]
    predicted_volatilities[stock] = predicted_volatility

def risk_parity_objective(weights, volatilities):
    risk_contributions = [vol * weight for vol, weight in zip(volatilities, weights)]
    total_portfolio_volatility = np.sum(risk_contributions)
    target_risk_contribution = 1 / len(volatilities)
    return sum([(rc/total_portfolio_volatility - target_risk_contribution)**2 for rc in risk_contributions])

initial_weights = [1/len(stock_list) for _ in stock_list]
bounds = [(0, 1) for _ in stock_list]
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
optimized = minimize(risk_parity_objective, initial_weights, args=(list(predicted_volatilities.values()),), method='SLSQP', bounds=bounds, constraints=constraints)

# 최적의 주식 비중
optimal_weights = optimized.x

# 실제 변동성 계산
actual_volatilities = {stock: np.std(returns[stock]) for stock in stock_data.keys()}

# 결과 딕셔너리 생성
results = {
    "stocks": stock_list,
    "predicted_volatilities": predicted_volatilities,
    "actual_volatilities": actual_volatilities,
    "optimal_weights": optimal_weights.tolist()
}

print(results)

from pprint import pprint

# 보낼 딕셔너리 값
pprint(results)

# 최적의 주식 비중 출력 및 설명
print("\n최적의 주식 비중:")

total_portfolio_volatility = sum([v * w for v, w in zip(list(predicted_volatilities.values()), optimal_weights)])
for stock, weight in zip(stock_list, optimal_weights):
    risk_contribution = predicted_volatilities[stock] * weight / total_portfolio_volatility
    print(f"{stock}: {weight:.2f} (리스크 기여도: {risk_contribution:.2%})")

    if weight < 0.01: # 비중이 매우 낮은 경우
        print(f"{stock}의 비중이 낮은 이유: 예상 변동성이 높기 때문에 포트폴리오 전체의 리스크를 줄이기 위해 이 주식의 비중을 낮추었습니다.")
    elif risk_contribution < (1/len(stock_list)) - 0.05: # 리스크 기여도가 평균보다 매우 낮은 경우
        print(f"{stock}의 리스크 기여도가 낮은 이유: 이 주식의 예상 변동성이 다른 주식에 비해 낮기 때문에 포트폴리오 전체의 리스크에 덜 기여합니다.")