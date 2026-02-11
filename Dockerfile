FROM python:3.11-slim

# set working directory
WORKDIR /app

# copy dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY app ./app

EXPOSE 8000

# run server with live reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
