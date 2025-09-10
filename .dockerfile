# use the ECR repo base-image link

FROM 886436929815.dkr.ecr.ap-southeast-1.amazonaws.com/skin-segment-app-base:3.9-slim-buster



WORKDIR /app



COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt &&\

    apt-get update -y &&\

    apt-get install curl -y 



COPY . .



EXPOSE 8501



HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health



# replace YourWebApp.py with the relevant file name

ENTRYPOINT ["streamlit", "run", "index.py", "--server.port=8501", "--server.address=0.0.0.0"]