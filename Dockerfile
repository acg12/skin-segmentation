# use the ECR repo base-image link

FROM 886436929815.dkr.ecr.ap-southeast-1.amazonaws.com/skin-segment-app-base:3.9-slim-buster



WORKDIR /app



RUN apt-get update -y && \
    apt-get install -y --no-install-recommends curl

# Now, copy and install Python dependencies.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY . .



EXPOSE 8501



HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health



# replace YourWebApp.py with the relevant file name

ENTRYPOINT ["streamlit", "run", "index.py", "--server.port=8501", "--server.address=0.0.0.0"]