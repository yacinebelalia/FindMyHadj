FROM python:3.11-slim




WORKDIR /App
COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential && \
    pip install --upgrade pip --no-cache-dir --default-timeout=100 -r requirements.txt


    

COPY . .
EXPOSE 8000

CMD ["uvicorn", "App.api:app", "--host", "0.0.0.0", "--port", "8000"]

