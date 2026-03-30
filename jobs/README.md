 # Labelling misinformation
The following job has it's own environment and requirements.
The app folder contains all the code for the job whilst the docs folder contains documentation on the job and general project infrastructure.
 
## Run
In order to run this app locally, a docker installation is required. Services are defined in the docker-compose.yml file where main logic is in the app service, a database can be deployed locally for testing, and tests can be run using the test service. There is also a testconsole service that allows you to run and test single commands:
```
docker compose up testconsole -d
docker compose exec testconsole bash
> pytest -vv -k test_pg_insert_data # will insert misinformation data linked to rmc on 2025 march 10
> exit
docker compose up app
```

### Secrets
 You have to set up your secrets (password, key etc.) inside your secrets folder
 ```
label_misinformation
├── app
├── docs
└── secrets
 ```

## Testing
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
The app config is setup via environment variables, here is an exhaustive list of the required environment variables needed:

* "MODEL_NAME" - the default model used inside openai 
* "MODEL_NAME_BRAZIL" - the model used inside openai
*  "APP_NAME" - will be used as folder inside bucket_output bucket
* "OPENAI_KEY"
*  "DATE" : to query a date with format YYYY-MM-DD, if none the date is yesterday
*  "BUCKET_OUTPUT" - bucket name with partitions : year,month,day,channel
*  "BUCKET_OUTPUT_BRAZIL" - bucket name with partitions : country,year,month,day,channel
*  "BUCKET_OUTPUT_BELGIUM" - bucket name with partitions : country,year,month,day,channel
* "MIN_MISINFORMATION_SCORE": 10 # the minimum score to have to be kept (10 out of 10)
* "CHANNEL" : mediatree former channel name (tf1 for TF1, itele for cnews, bfmtv for BFMTv ...)
* NUMBER_OF_PREVIOUS_DAYS (integer): default 7 days to check if something missing - in case production servers had an issue
* COUNTRY: The country or country collection for which to run the job. by default this should be `prod`. Possible values are france, brazil, belgium, all (france, brazil and belgium) and prod (france and brazil). For more details see label_misinformation/app/country.py.

### To sync data after each process
This can be done manually by clicking on UI, but if you set this it will be done after each cron job if LABEL_STUDIO_PROJECT_ID is set :
* LABEL_STUDIO_URL = container url
* LABEL_STUDIO_PROJECT_ID = Storage ID (not project id) inside label studio --> new to see labelstudio logs when cliking on "sync" it can be different from the one inside the label interface. @see https://api.labelstud.io/api-reference/api-reference/import-storage/s-3/sync ( [2025-05-06 12:41:25,789] [django.server::log_message::213] [INFO] "POST /api/storages/s3/5/sync HTTP/1.1" 200 799 )
* LABEL_STUDIO_PROJECT = The actual project ID
* LABEL_STUDIO_PROJECT_ID_BRAZIL = Storage ID for the brazil project
* LABEL_STUDIO_PROJECT_BRAZIL = Project ID for the brazil project
* API_LABEL_STUDIO_KEY = label studio API key

### To reparse
* LABEL_MISINFORMATION_OVERWRITE = Boolean. If activated the job will reparse any extract that has not been included in labelstudio already and add save it to s3.

## Deployment

App image is pushed on the Scaleway Container Registry, and then deployed on the Scaleway Serverless service.

For production, to verify no days have been forgotten, NUMBER_OF_PREVIOUS_DAYS is used to check it. In case a day is missing it's recalculated.

### Bump
```
bump2version patch  # 0.1.0 → 0.1.1
bump2version minor  # 0.1.0 → 0.2.0
bump2version major  # 0.1.0 → 1.0.0
```
