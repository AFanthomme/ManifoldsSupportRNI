FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
WORKDIR /app
# RUN groupadd -r appgroup && useradd -r -g appgroup appuser
# RUN chown -R appuser:appgroup /app 
# USER appuser
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./src ./src
