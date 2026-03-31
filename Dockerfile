FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces passes secrets as environment variables at runtime.
# Declare them here so they're visible to the container.
# You set the actual values in: HF Space → Settings → Variables and secrets
ENV API_BASE_URL=http://localhost:7860
ENV MODEL_NAME=llama-3.3-70b-versatile
ENV GROQ_API_KEY_1=""
ENV GROQ_API_KEY_2=""
ENV GROQ_API_KEY_3=""
ENV GROQ_API_KEY_4=""
ENV HF_TOKEN=""

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
