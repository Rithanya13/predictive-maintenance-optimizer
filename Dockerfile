FROM python:3.11-slim

WORKDIR /app

# System deps for xgboost/shap (libgomp for OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model and precompute the analyses at build time so the image is
# self-contained and the API starts with ready artifacts, rather than
# computing them lazily on someone's first request.
RUN python3 src/train_model.py \
    && python3 src/segment_analysis.py \
    && python3 src/unsupervised_analysis.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
