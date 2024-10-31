FROM svizor/zoomcamp-model:3.11.5-slim

# Actualiza repositorios
RUN apt update

# instala pipenv

RUN pip install pipenv
RUN apt install -y pipenv

# Copia los archivos de dependencias
WORKDIR /app
COPY Pipfile* /app/

# instala las dependencias pero a nivel sistema
RUN pipenv install --system --deploy --ignore-pipfile

# Copia los archivos del proyecto
COPY sub_serving.py /app/

# Exponer el puerto 
EXPOSE 9696

# Especificar punto de entrada
ENTRYPOINT gunicorn --bind=0.0.0.0:9696 sub_serving:app