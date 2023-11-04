# 뉴스 제목, 일자 크롤링

import requests
from bs4 import BeautifulSoup
import datetime
import time
import pandas as pd


# 개선할 점 -> 중간에 부정/긍정 단어를 만들어서 딕셔너리 만든 다음에 크롤링해서 감성분석한거랑 매칭해야할 것 같음 !!!!!
# 크롤링 시간이 너무 많이 걸림



base_url = "https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=101&date="

# 1. User-Agent를 설정
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 시작 날짜와 종료 날짜 설정
start_date = datetime.datetime.now() - datetime.timedelta(days=30)
end_date = datetime.datetime.now() - datetime.timedelta(days=5)

current_date = start_date

news_data = []

while current_date <= end_date:
    # 2. 요청을 보낼 때 headers 추가
    response = requests.get(base_url + current_date.strftime('%Y%m%d'), headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    for item in soup.select(".cluster_text a"):
        title = item.text.strip()
        if item.attrs["href"].startswith("http"):
            news_url = item.attrs["href"]
        else:
            news_url = "https:" + item.attrs["href"]

        detail_response = requests.get(news_url, headers=headers)  # headers 추가
        detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
        date_element = detail_soup.select_one("span.media_end_head_info_datestamp_time")
        date = date_element.attrs["data-date-time"].split()[0]

        news_data.append({
            'title': title,
            'date': date
        })

        # 요청 간에 약간의 지연을 두어 IP 차단을 피하기
        time.sleep(1.5)

    # 다음 날짜로 이동
    current_date += datetime.timedelta(days=1)

# 뉴스 데이터를 날짜 순으로 정렬
news_data_sorted = sorted(news_data, key=lambda x: x['date'])


# 데이터프레임으로 변환
news_df = pd.DataFrame(news_data_sorted, columns=['date', 'title'])

# 시작 날짜와 종료 날짜를 문자열로 변환
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# 원하는 날짜 범위만 선택
news_df = news_df[(news_df['date'] >= start_date_str) & (news_df['date'] <= end_date_str)]


# 크롤링한 뉴스 제목 명사 추출

from konlpy.tag import Okt

# Okt 객체 초기화
okt = Okt()

# 제목에서 명사만 추출하는 함수
def extract_nouns(title):
    return ', '.join(okt.nouns(title))

# 'title' 열의 각 제목에 대하여 명사만 추출
news_df['nouns'] = news_df['title'].apply(extract_nouns)

print(news_df)

# 코스피 등락율 - 뉴스

import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_kospi_closing_prices():
    url = "https://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    kospi_closings = []

    for i in range(1, 7):
        response = requests.get(url + f"&page={i}", headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        dates = soup.select(".date")
        closings = soup.select(".number_1")

        for d, c in zip(dates, closings[::4]):  # 종가만 가져오기 위해 slicing 사용
            kospi_closings.append([d.text.strip(), float(c.text.replace(',', ''))])

    return kospi_closings

kospi_data = get_kospi_closing_prices()
df = pd.DataFrame(kospi_data, columns=["Date", "Closing"])

# Shift를 사용해 다음 날짜의 종가를 가져와서 현재 날짜와 비교
# 예: 10월 9일 종가보다 10월 10일 종가가 더 높다면 10월 9일 등락율 1이 됨

df["Up/Down"] = (df["Closing"].shift(1) > df["Closing"]).astype(int)

# 날짜 기준으로 최근 30일의 데이터를 가져온 후, 정렬
df = df.sort_values(by="Date").tail(30).reset_index(drop=True)

df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d').dt.strftime('%Y-%m-%d')

print(df)

merged_df = pd.merge(news_df, df, left_on='date', right_on='Date', how='inner')
merged_df = merged_df[['date', 'Up/Down', 'title', 'nouns']]
print(merged_df)

# 단어 점수 초기화 및 개수
# 한 글자 제외 -> 두 글자 이상 단어

# 'nouns' 칼럼의 값을 문자열로 변환
df['nouns'] = merged_df['nouns'].astype(str)
df = df.dropna(subset=['nouns'])

# 'filtered_nouns' 컬럼 생성
merged_df['filtered_nouns'] = merged_df['nouns'].apply(lambda x: [word for word in x.split(', ') if len(word) > 1])

# 단어 점수 초기화
word_scores = {word: 0 for word_list in merged_df['filtered_nouns'] for word in word_list}
print(word_scores)

# 단어 빈도수 계산
from collections import Counter
word_counts = Counter(word for word_list in merged_df['filtered_nouns'] for word in word_list)

# 결과 출력
for word, count in word_counts.items():
    print(f"{word}: {count}")

# 단어 점수 부여

# 전체 단어 개수 출력
total_words = sum(word_counts.values())
print(f"\nTotal number of words: {total_words}")

# Up/Down 값이 1인 데이터에서 포함된 단어의 리스트
up = []
for nouns in merged_df[merged_df['Up/Down'] == 1]['filtered_nouns']:
    up.extend(nouns)

# Up/Down 값이 0인 데이터에서 포함된 단어의 리스트
down = []
for nouns in merged_df[merged_df['Up/Down'] == 0]['filtered_nouns']:
    down.extend(nouns)

print("up :", len(up))
print("down :", len(down))

# 상승 비율과 하락 비율 계산
total_words = len(up) + len(down)
up_ratio = len(up) / total_words
down_ratio = len(down) / total_words

# 단어 점수 초기화
word_scores = {word: 0 for word in word_scores.keys()}  # 기존의 word_scores 딕셔너리 사용

# Up(1) 데이터의 단어들에 대해서 하락 비율을 더해주기
for word in up:
    if word in word_scores:
        word_scores[word] += down_ratio

# Down(0) 데이터의 단어들에 대해서 상승 비율을 차감해주기
for word in down:
    if word in word_scores:
        word_scores[word] -= up_ratio

# 결과 확인
print(word_scores)

# 감성 사전 완료

total = []
for nouns in merged_df['filtered_nouns']:
    sent_score = 0
    for noun in nouns:
        if noun in word_scores:
            sent_score += word_scores[noun]

    # 해당 뉴스 제목에 포함된 단어의 수로 나누어 평균 점수를 계산
    avg_sent_score = sent_score / len(nouns) if nouns else 0  # 단어가 없는 경우 0으로 처리
    total.append(avg_sent_score)

merged_df['sent_score'] = total

def calculate_sentiment_score(noun_list):
    score = 0
    for noun in noun_list:
        if noun in word_scores:  # word_scores는 기존에 구한 감성 사전입니다.
            score += word_scores[noun]
    return score / (len(noun_list) if len(noun_list) != 0 else 1)



# 감성사전의 평균 점수 계산

sent_mean = sum(word_scores.values()) / len(word_scores)
print('감성 사전 평균 점수 : ',sent_mean)

# 감성 점수 계산
def calculate_sentiment_score(noun_list):
    score = 0
    for noun in noun_list:
        if noun in word_scores:
            score += word_scores[noun]
    return score / (len(noun_list) if len(noun_list) != 0 else 1)

merged_df['sent_score'] = merged_df['filtered_nouns'].apply(calculate_sentiment_score)

# 평균 점수를 기준으로 라벨링
merged_df['sent_label'] = merged_df['sent_score'].apply(lambda x: 1 if x > sent_mean else 0)



result_df = merged_df[['date', 'Up/Down', 'sent_score', 'sent_label', 'title', 'nouns']]
print(result_df)

# 모델링

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 1. 데이터 준비
X_data = merged_df['nouns'].apply(lambda x: ' '.join([word for word in x.split(', ') if len(word) > 1])).values
Y_data = merged_df['sent_label'].values  # 기존의 'merged_df'를 사용

# 2. 토큰화 및 패딩
vocab_size = 2000
tokenizer = Tokenizer(num_words=vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(X_data)
X_tokenized = tokenizer.texts_to_sequences(X_data)
X_padded = pad_sequences(X_tokenized, maxlen=30)

# 3. Bi-LSTM 모델 구축 및 훈련
model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = model.fit(X_padded, Y_data, epochs=15, callbacks=[es, mc], batch_size=256, validation_split=0.2)

# 최신 뉴스 크롤링 후 모델에 적용해서 최종 결과 확

# 뉴스 제목 크롤링
base_url = "https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=101&date="
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

start_date = datetime.datetime.now() - datetime.timedelta(days=30)
end_date = datetime.datetime.now() - datetime.timedelta(days=5)
current_date = start_date

news_data = []

while current_date <= end_date:
    response = requests.get(base_url + current_date.strftime('%Y%m%d'), headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    for item in soup.select(".cluster_text a"):
        title = item.text.strip()
        news_data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'title': title
        })
    current_date += datetime.timedelta(days=1)

news_df = pd.DataFrame(news_data, columns=['date', 'title'])

# 명사 추출
okt = Okt()
news_df['nouns'] = news_df['title'].apply(lambda x: ', '.join(okt.nouns(x)))

# 'filtered_nouns' 컬럼 생성
news_df['filtered_nouns'] = news_df['nouns'].apply(lambda x: [word for word in x.split(', ') if len(word) > 1])

# 감성 사전과 비교하여 감성 점수 계산
news_df['filtered_nouns'] = news_df['nouns'].apply(lambda x: [word for word in x.split(', ') if len(word) > 1])
news_df['sent_score'] = news_df['filtered_nouns'].apply(calculate_sentiment_score)  # 이전에 정의한 함수

# 예측을 위한 데이터 전처리
X_test_tokenized = tokenizer.texts_to_sequences(news_df['nouns'].apply(lambda x: ' '.join([word for word in x.split(', ') if len(word) > 1])).values)
X_test_padded = pad_sequences(X_test_tokenized, maxlen=30)

# 훈련된 Bi-LSTM 모델로 예측
predicted = model.predict(X_test_padded)
news_df['predicted_label'] = (predicted > 0.5).astype(int)

# 결과 출력
positive_news_ratio = news_df['predicted_label'].sum() / len(news_df)
if positive_news_ratio > 0.5:
    print("미래 주식 시장은 긍정적으로 움직일 것으로 예상됩니다.")
else:
    print("미래 주식 시장은 부정적으로 움직일 것으로 예상됩니다.")

    # 각 뉴스의 감성 점수를 바탕으로 긍정적 및 부정적 뉴스 개수 확인 및 출력
sent_mean = news_df['sent_score'].mean()

pos_news = len(news_df[news_df['sent_score'] > sent_mean])
neg_news = len(news_df[news_df['sent_score'] <= sent_mean])
total_news = len(news_df)

print(f"긍정적 뉴스 수: {pos_news} ({pos_news/total_news*100:.2f}%)")
print(f"부정적 뉴스 수: {neg_news} ({neg_news/total_news*100:.2f}%)")

print("\n긍정적 뉴스 예시:")
for title in news_df[news_df['sent_score'] > sent_mean]['title'].head(5):
    print("-", title)

print("\n부정적 뉴스 예시:")
for title in news_df[news_df['sent_score'] <= sent_mean]['title'].head(5):
    print("-", title)