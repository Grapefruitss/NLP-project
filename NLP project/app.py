from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
import pickle
import re

app = FastAPI()

app.mount("/static", StaticFiles(directory='static'), name='static')

templates = Jinja2Templates(directory='templates')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class textRequest(BaseModel):
    text: str

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(request: textRequest):
    # 셀레니움을 사용하여 파파고 번역 수행
    translated_paragraph = translate(request.text)

    # 자소서 정규화
    regularized_paragraph = regularize(translated_paragraph)
    
    # 자소서 토큰화
    stop_words = set(stopwords.words('english'))
    new_tokens = word_tokenize(regularized_paragraph)
    filtered_tokens = [word for word in new_tokens if word.isalpha() and word not in stop_words]

    # 새 자소서의 벡터 표현 생성
    new_vector = model.infer_vector(filtered_tokens)

    # 모델에 입력값 전달하고 결과 예측
    predicted_result = model.predict(new_vector)

    return {'predicted-result': predicted_result}

def translate(text):
    # 크롬 드라이버 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # 파파고 웹사이트 열기
    papago_url = "https://papago.naver.com/"
    driver.get(papago_url)

    # 페이지 로딩 시간 대기
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "textarea#txtSource")))

    # 번역할 텍스트 입력
    if len(text) >= 3000:
        dot_index = text[3000:2500:-1].index('.')
        x1, x2 = text[:3000-dot_index+1], text[3000-dot_index+1:]
        input_text = x1  # 번역할 텍스트
        input_box = driver.find_element(By.CSS_SELECTOR, "textarea#txtSource")
        input_box.clear()
        input_box.send_keys(input_text)
        time.sleep(20)
    
        translate_button = driver.find_element(By.CSS_SELECTOR, "button#btnTranslate")
        translate_button.click()
    
        # 번역 결과 대기
        output_element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#txtTarget")))
    
        # 번역 결과 추출
        output = output_element.text
        translated_x1 = output

        input_text = x2  # 번역할 텍스트
        input_box = driver.find_element(By.CSS_SELECTOR, "textarea#txtSource")
        input_box.clear()
        input_box.send_keys(input_text)
        time.sleep(20)
    
        translate_button = driver.find_element(By.CSS_SELECTOR, "button#btnTranslate")
        translate_button.click()
    
        # 번역 결과 대기
        output_element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#txtTarget")))
    
        # 번역 결과 추출
        output = output_element.text
        translated_x2 = output
        translated_essay = translated_x1 + translated_x2

    else:
        input_text = text  # 번역할 텍스트
        input_box = driver.find_element(By.CSS_SELECTOR, "textarea#txtSource")
        input_box.clear()
        input_box.send_keys(input_text)
        time.sleep(20)
    
        translate_button = driver.find_element(By.CSS_SELECTOR, "button#btnTranslate")
        translate_button.click()
    
        # 번역 결과 대기
        output_element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#txtTarget")))
    
        # 번역 결과 추출
        translated_essay = output_element.text
    
    driver.quit()

    return translated_essay

def regularize(text):
    re_text = text.lower()                               # 소문자화
    re_text = re.sub(r'\W', ' ', re_text)                # 알파벳, 숫자 제외 한 특수문자 제거
    re_text = re.sub(r'\d', ' ', re_text)                # 숫자 제거
    re_text = re.sub(r'_', ' ', re_text)                 # '_' 제거
    re_text = re.sub(r'\s+', ' ', re_text)               # 잉여 space 제거.
    re_text = re.sub(",|\n|@|:", "", re_text)            # 쉼표, \n, @ 제거
    re_text = re.sub(r'[\[\]\(\)\{\}]', '', re_text)     # 괄호 제거
    re_text = re.sub(r'^\s+', '', re_text)               # 공백으로 시작하는 문서들의 공백 제거
    return re_text