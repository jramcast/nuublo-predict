FROM python:3.5-slim

# Create app directory
RUN mkdir -p /opt/nuublo-predict
WORKDIR /opt/nuublo-predict

# Install app dependencies
COPY requirements.txt /opt/nuublo-predict
RUN pip install -r requirements.txt

# Download corporas
RUN python -m nltk.downloader stopwords && \
    python -m textblob.download_corpora

# Bundle app source
COPY . /opt/nuublo-predict

EXPOSE 8752
CMD [ "python", "server.py" ]
