resource "scaleway_job_definition" "main" {
  name         = "testjob"
  cpu_limit    = 4000
  memory_limit = 8192
  image_uri    = "rg.fr-par.scw.cloud/misinformation/label-misinformation:latest"
  timeout      = "60m"
  project_id   = scaleway_account_project.project_climatesafeguards.id

  cron {
    schedule = "5 6 * * *"
    timezone = "Europe/Paris"
  }

  env = {
    "APP_NAME" = "label-misinformation-${var.country}-${var.environment}"

    "AWS_DEFAULT_REGION" = "fr-par"

    "BUCKET_INPUT"         = "mediatree"
    "BUCKET_OUTPUT"        = scaleway_object_bucket.source_bucket.name
    "BUCKET_OUTPUT_FOLDER" = "label-misinformation-input"
    "COUNTRY"              = var.country

    "ENV" = "production" # maybe this should be equal to var.environment.. does not work for now

    "KEYWORDS_URL"            = "https://keywords.mediatree.fr/api/export/single"
    "LABEL_STUDIO_PROJECT"    = var.labelstudio_project
    "LABEL_STUDIO_PROJECT_ID" = var.labelstudio_project_id
    "LABEL_STUDIO_URL"        = scaleway_container.labelstudio_container.domain_name
    "LOGLEVEL"                = "INFO"
    "MODEL_NAME"              = var.model_name
    "MODIN_MEMORY"            = 3000000000
    "POSTGRES_DB"             = data.scaleway_rdb_database.barometre.name
    "POSTGRES_HOST"           = data.scaleway_rdb_instance.barometre_rdb.endpoint_ip
    "POSTGRES_PORT"           = data.scaleway_rdb_instance.barometre_rdb.endpoint_port
    "POSTGRES_USER"           = scaleway_rdb_user.barometre_user.name
    "RAY_COLOR_PREFIX"        = 1
    "SENTRY_DSN"              = "https://59b6ac362c8725c76e725438483ae87d@o4506785184612352.ingest.us.sentry.io/4508846066630656"

  }

  secret_reference {
    secret_id   = scaleway_secret.secret_label_studio_token.id
    environment = "API_LABEL_STUDIO_KEY"
  }
  secret_reference {
    secret_id   = scaleway_secret.access_key_id.id
    environment = "BUCKET_ACCESS"
  }
  secret_reference {
    secret_id   = scaleway_secret.secret_key_id.id
    environment = "BUCKET_SECRET"
  }
  secret_reference {
    secret_id   = scaleway_secret.mediatree_password.id
    environment = "MEDIATREE_PASSWORD"
  }
  secret_reference {
    secret_id   = scaleway_secret.openai_api_key.id
    environment = "OPENAI_API_KEY"
  }
  secret_reference {
    secret_id   = scaleway_secret.postgres_password_barometre.id
    environment = "POSTGRES_PASSWORD"
  }
  depends_on = [scaleway_container.labelstudio_container]
}
