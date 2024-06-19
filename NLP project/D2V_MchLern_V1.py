import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize

#################################################### 데이터 준비 및 전처리

# 데이터 로드
data = pd.read_csv('cover_letters.csv')  # 자소서 : 'text' // 합격, 불합격 :  'label' (0, 1)

# 텍스트와 라벨 분리
texts = data['text'].astype(str).tolist()
labels = data['label'].tolist()

# 라벨 인코딩
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

#################################################### Doc2Vec 모델 학습

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 학습 데이터를 태그된 문서로 변환
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(X_train)]

# Doc2Vec 모델 설정 및 학습
doc2vec_model = Doc2Vec(
    vector_size=100,  # 벡터 차원 수
    window=5,         # 컨텍스트 윈도우 크기
    min_count=2,      # 최소 출현 빈도
    workers=4,        # 사용 스레드 수
    epochs=40,        # 학습 반복 횟수
    dm=1              # DM 모델 사용 (0이면 DBOW 모델 사용)
)

# 모델 학습
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

####################################################### 벡터화 및 분류 모델 학습

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 학습 데이터 벡터화
X_train_vectors = [doc2vec_model.infer_vector(word_tokenize(doc.lower())) for doc in X_train]
X_test_vectors = [doc2vec_model.infer_vector(word_tokenize(doc.lower())) for doc in X_test]

# 분류 모델 학습
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_vectors, y_train)

# 예측
y_pred = classifier.predict(X_test_vectors)

# 평가
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

######################################################## Grid Search를 사용하여 랜덤 포레스트 분류기의 하이퍼파라미터를 튜닝

from sklearn.model_selection import GridSearchCV

# 랜덤 포레스트 분류기 초기화
rf = RandomForestClassifier(random_state=42)

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [100, 200, 300], # 랜덤 포레스트에서 생성할 트리의 개수입니다. 100, 200, 300개의 트리를 사용하여 각각의 경우를 테스트합니다.
    'max_depth': [10, 20, 30], # 각 트리의 최대 깊이입니다. 트리의 최대 깊이를 10, 20, 30으로 설정하여 각각의 경우를 테스트합니다.
    'min_samples_split': [2, 5, 10], # 내부 노드를 분할하는 데 필요한 최소 샘플 수입니다. 이 값을 2, 5, 10으로 설정하여 각각의 경우를 테스트합니다.
    'min_samples_leaf': [1, 2, 4], # 리프 노드에 있어야 하는 최소 샘플 수입니다. 이 값을 1, 2, 4로 설정하여 각각의 경우를 테스트합니다.
    'max_features': ['sqrt', 'log2'] #  각 트리를 분할할 때 고려할 최대 특징 수입니다. 제곱근(sqrt)과 로그(log2) 값을 사용하여 각각의 경우를 테스트합니다.
}

# Grid Search 설정
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=3,  # 교차 검증 3번
                           n_jobs=-1,  # 가능한 모든 프로세서를 사용하여 병렬 처리를 수행
                           verbose=2)  # 진행 상황을 출력

# 학습 데이터로 Grid Search 수행
grid_search.fit(X_train_vectors, y_train)

# 최적의 하이퍼파라미터 출력
print("Best parameters found: ", grid_search.best_params_)

# 최적의 모델로 테스트 데이터 예측
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_vectors)

# 성능 평가
print(classification_report(y_test, y_pred))

################################################### Random Search를 이용하여 하이퍼파라미터를 튜닝

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# 랜덤 포레스트 분류기 초기화
rf = RandomForestClassifier(random_state=42)

# 하이퍼파라미터 분포 설정
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2']
}

# Random Search 설정
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=3, n_jobs=-1, random_state=42, verbose=2)

# 학습 데이터로 Random Search 수행
random_search.fit(X_train_vectors, y_train)

# 최적의 하이퍼파라미터 출력
print("Best parameters found: ", random_search.best_params_)

# 최적의 모델로 테스트 데이터 예측
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test_vectors)

# 성능 평가
print(classification_report(y_test, y_pred))

################################################### 새로운 데이터 예측

new_text = "새로운 자기소개서 텍스트를 여기에 입력합니다."
new_vector = doc2vec_model.infer_vector(word_tokenize(new_text.lower()))
prediction = classifier.predict([new_vector])
predicted_label = label_encoder.inverse_transform(prediction)
print(f"예측 결과: {predicted_label[0]}")
