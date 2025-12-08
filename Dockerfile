# Python 3.10 기반 슬림 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 1. 필수 라이브러리 목록을 정의합니다.
# requirements.txt 파일이 없으므로 직접 나열합니다.
RUN echo "Flask" > requirements.txt && \
    echo "gunicorn" >> requirements.txt && \
    echo "onnxruntime==1.17.0" >> requirements.txt && \
    echo "Pillow" >> requirements.txt && \
    echo "numpy" >> requirements.txt && \
    echo "mysql-connector-python" >> requirements.txt && \
    echo "flask-cors" >> requirements.txt

# 2. 종속성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 3. 애플리케이션 코드와 모델 디렉토리 복사
# app.py와 DB 관련 함수들이 있는 파일을 app.py라고 가정
COPY apiv3.py .
# model/ 디렉토리 전체를 복사합니다.
COPY model /app/model

# 4. 포트 8000 노출
EXPOSE 8000

# 5. 엔트리포인트: Gunicorn을 사용하여 Flask 앱 실행
# app.py 파일 내의 Flask 앱 인스턴스 변수 이름이 'app'이라고 가정
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]