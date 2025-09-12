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
docker compose up labelstudio
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
docker compose up testconsole -d
docker compose exec testconsole bash
> pytest -vv -k test_pg_insert_data # will insert misinformation data linked to rmc on 2025 march 10
> exit
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
* env variable "MODEL_NAME" - the default model used inside openai (used for france mainly)
"* env variable MODEL_NAME_BRAZIL" - the model used inside openai
* env variable  "APP_NAME" - will be used as folder inside bucket_output bucket
* env variable "OPENAI_KEY"
* env variable  "DATE" : to query a date with format YYYY-MM-DD, if none the date is yesterday
* env variable  "BUCKET_OUTPUT" - bucket name with partitions : year,month,day,channel
* env variable  "BUCKET_OUTPUT_BRAZIL" - bucket name with partitions : country,year,month,day,channel
* env variable  "BUCKET_OUTPUT_BELGIUM" - bucket name with partitions : country,year,month,day,channel
* env variable "MIN_MISINFORMATION_SCORE": 10 # the minimum score to have to be kept (10 out of 10)
* env variable : "CHANNEL" : mediatree former channel name (tf1 for TF1, itele for cnews, bfmtv for BFMTv ...)
* env variable : NUMBER_OF_PREVIOUS_DAYS (integer): default 7 days to check if something missing - in case production servers had an issue
* env variable COUNTRY: The country or country collection for which to run the job. by default this should be `prod`. Possible values are france, brazil, belgium, all (france, brazil and belgium) and prod (france and brazil). For more details see label_misinformation/app/country.py.

### To sync data after each process
This can be done manually by clicking on UI, but if you set this it will be done after each cron job if LABEL_STUDIO_PROJECT_ID is set :
* env variable : LABEL_STUDIO_URL = container url
* env variable : LABEL_STUDIO_PROJECT_ID = Storage ID (not project id) inside label studio --> new to see labelstudio logs when cliking on "sync" it can be different from the one inside the label interface. @see https://api.labelstud.io/api-reference/api-reference/import-storage/s-3/sync ( [2025-05-06 12:41:25,789] [django.server::log_message::213] [INFO] "POST /api/storages/s3/5/sync HTTP/1.1" 200 799 )
* env variable : LABEL_STUDIO_PROJECT = The actual project ID
* env varuable : LABEL_STUDIO_PROJECT_ID_BRAZIL = Storage ID for the brazil project
* env varuable : LABEL_STUDIO_PROJECT_BRAZIL = Project ID for the brazil project
* env variable : API_LABEL_STUDIO_KEY = label studio API key

### To reparse
* env variable : LABEL_MISINFORMATION_OVERWRITE = Boolean. If activated the job will reparse any extract that has not been included in labelstudio already and add save it to s3.

## Deployment

App image is pushed on the Scaleway Container Registry, and then deployed on the Scaleway Serverless service.

For production, to verify no days have been forgotten, NUMBER_OF_PREVIOUS_DAYS is used to check it. In case a day is missing it's recalculated.

### Bump
```
bump2version patch  # 0.1.0 → 0.1.1
bump2version minor  # 0.1.0 → 0.2.0
bump2version major  # 0.1.0 → 1.0.0
```
