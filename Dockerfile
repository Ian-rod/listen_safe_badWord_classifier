FROM python:3.12.3-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip --timeout 1000

RUN pip install --no-cache-dir -r requirements.txt --timeout 1000

COPY classifierAPI.py .
COPY modelInterface.py .
COPY model.pkl .
COPY vectorizer.pkl .

EXPOSE 8000

CMD ["uvicorn", "classifierAPI:app", "--host", "0.0.0.0", "--port", "8000","--reload"]

