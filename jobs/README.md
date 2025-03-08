 # Labelling misinformation jobs

 
 ## Going to production from a jupyter notebook

 ### Init - create the python file
 If the raw python file is jupyter notebook, convert it to a python file using :
 ```
 sudo apt install jupyter-nbconvert
 jupyter nbconvert *.ipynb --to python
 # delete .ipynb file
```

## Label Studio locally
```
docker compose up label_studio
```

## Build
```
python3 -m venv venv
source venv/bin/activate  #.fish if using fish
pip freeze > requirements.txt
```

## Run
Install [Docker](https://www.docker.com/get-started/) in a few minutes on the official website then :
```
docker compose up app
```

Not working ? You have to set up your secrets (password, key etc.) inside your secrets folder

## Test
Tests are located inside tests folder
### Run all test
```
docker compose up test
```

### Target one test
```
docker compose up testconsole -d
docker compose exec testconsole bash
> pytest --log-level INFO -vv -k s3_utils 
> pytest --log-level INFO -vv -k get_new_plaintext_from_whisper_mp4
```

## Configuration
* env variable "MODEL_NAME" - the model used inside openai
* env variable  "APP_NAME" - will be used as folder inside bucket_output bucket
* env variable "OPENAI_KEY"
* env variable  "DATE" : to query a date with format YYYY-MM-DD, if none the date is yesterday
* env variable  "BUCKET_INPUT" - bucket name with partitions : year,month,day,channel
* env variable  "BUCKET_OUTPUT" - bucket name with partitions : year,month,day,channel
* env variable "MIN_MISINFORMATION_SCORE": 10 # the minimum score to have to be kept (10 out of 10)
* env variable : "CHANNEL" : mediatree former channel name (tf1 for TF1, itele for cnews, bfmtv for BFMTv ...)
* env variable : NUMBER_OF_PREVIOUS_DAYS (integer): default 7 days to check if something missing - in case production servers had an issue


## Deployment

App image is pushed on the Scaleway Container Registry, and then deployed on the Scaleway Serverless service.

For production, to verify no days have been forgotten, NUMBER_OF_PREVIOUS_DAYS is used to check it. In case a day is missing it's recalculated.

### Bump
```
bump2version patch  # 0.1.0 → 0.1.1
bump2version minor  # 0.1.0 → 0.2.0
bump2version major  # 0.1.0 → 1.0.0
```
