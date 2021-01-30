FROM python:3.8

# le dossier /code sera le dossier courant du container
WORKDIR /Projet-Cloud-Computing

# Copy du fichier dans le container
COPY requirements.txt .

# installation des bibliothéques spécifié dans le fichier
RUN pip install -r requirements.txt

COPY ml_template_api/ .

# commande a executer au lancement du container
CMD [ "python", "ml/model.py --model=linearRegression --split=0.1" ]
CMD [ "uvicorn", "uvicorn main:app --reload" ]