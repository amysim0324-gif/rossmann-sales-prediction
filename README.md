# Rossmann 매출 예측 앱 — 실행 방법

## 1. 이 폴더 전체를 맥 컴퓨터로 다운로드
`rossmann_app` 폴더 안의 4개 파일이 모두 같은 폴더에 있어야 해요:
- app.py
- rossmann_model.json
- feature_columns.json
- store_info.csv
- requirements.txt

## 2. 터미널(맥의 터미널 앱)을 열고, 이 폴더로 이동

```bash
cd 다운로드받은_폴더_경로/rossmann_app
```

## 3. 필요한 라이브러리 설치 (최초 1회만)

```bash
pip install -r requirements.txt
```

## 4. 앱 실행

```bash
streamlit run app.py
```

실행하면 자동으로 브라우저 창이 열리고, `http://localhost:8501` 같은 주소로
웹앱이 뜰 거예요. 이게 채은님 컴퓨터 안에서만 열리는 로컬 웹페이지예요.

## 5. 포트폴리오용으로 온라인에 공개하고 싶다면

지금은 로컬(내 컴퓨터)에서만 열리는 상태예요. 누구나 링크로 접속하게
하려면 **Streamlit Community Cloud**(무료)에 배포하면 돼요 — 이건 다음
단계에서 같이 진행하면 돼요.
