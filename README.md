# climateguard

The repo contains the code for an AI media screening application. This service pulls data from the OME database, checks it with a simple prompt and gives a misinformation probability score. Scores >= 8 are flagged and saved on an object storage in order to be synced with labelstudio, that is here used as a journalist annotation platform.

## Serveless Jobs
The main logic of the service is deployed as a scaleway serverless job. You can find all serverless jobs in the jobs folder. The main logic is contained in the label-misinformation folder. 

## IaaC
The project is templatised to be deployed for multiple countries, on multiple subjects. The OpenTofu code is found in the infrastructure folder. Refer to `infrastructure/Readme.md` for details.

## Labelstudio Interfaces
The labelstudio interfaces are defined by xml files living in the `deployments/labelstudio` folder. 

## Data
The notebooks used to create the dataset by pulling records from labelstudio and then analyse it live in the `data` folder  

## Models
Scripts for finetuning models and for experimenting with different techiniques are in the `climateguard` folder under `climateguard/finetuning` and `climateguard/experiments` respectively.

# Getting started
Install dependencies with the `setup.sh` script.
