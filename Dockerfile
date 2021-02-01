FROM python:3.8

COPY ml_template_api/ .

# le dossier /code sera le dossier courant du container
WORKDIR /ml_template_api

# Copy du fichier dans le container
COPY requirements.txt .

# installation des bibliothéques spécifié dans le fichier
RUN pip install -r requirements.txt



# commande a executer au lancement du container
#CMD [ "unicorn", "./ml/model.py --model=linearRegression --split=0.1" ]
RUN python /ml/model.py --model=linearRegression --split=0.1

CMD ["uvicorn", "main:app"]
