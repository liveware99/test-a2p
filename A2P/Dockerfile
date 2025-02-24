# Use an official Conda image as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /A2P

# Copy the current directory contents into the container at /app
COPY . /A2P

# Create a Conda environment named 'a2p' with Python version 3.11
RUN conda create --name a2p python=3.11

# Activate the Conda environment
RUN echo "source activate a2p" > ~/.bashrc
ENV PATH /opt/conda/envs/a2p/bin:$PATH

# Install dependencies from Conda
RUN conda install -c conda-forge sentence-transformers=2.5.1
RUN conda install -c conda-forge fastapi=0.110.0
# RUN conda install -c conda-forge pandas
RUN conda install -c conda-forge apscheduler=3.10.4 
RUN conda install -c conda-forge sqlalchemy=2.0.28
RUN conda install -c conda-forge uvicorn=0.28.0
RUN conda install -c conda-forge psutil=5.9.8
RUN conda install -c conda-forge aiomysql=0.2.0 
# RUN conda install -c conda-forge pymysql
# RUN pip install aiomysql
