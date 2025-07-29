resource "scaleway_job_definition" "main" {
  name         = "label-misinformation-${var.country}-${var.environment}"
  cpu_limit    = 4000
  memory_limit = 8192
  image_uri    = "rg.fr-par.scw.cloud/misinformation/label-misinformation:latest"
  timeout      = "60m"
  project_id   = data.scaleway_account_project.project.id

  cron {
    schedule = var.job_cron_schedule
    timezone = var.job_cron_schedule_timezone
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
    "LABEL_STUDIO_URL"        = "https://${scaleway_container.labelstudio_container.domain_name}"
    "LOGLEVEL"                = "INFO"
    "MEDIATREE_AUTH_URL"      = "https://keywords.mediatree.fr/api/auth/token/"
    "MEDIATREE_USER"          = "quotaclimat"
    "MODEL_NAME"              = var.model_name
    "MODIN_MEMORY"            = 3000000000
    "POSTGRES_DB"             = data.scaleway_rdb_database.barometre.name
    "POSTGRES_HOST"           = data.scaleway_rdb_instance.barometre_rdb.endpoint_ip
    "POSTGRES_PORT"           = data.scaleway_rdb_instance.barometre_rdb.endpoint_port
    "POSTGRES_USER"           = "barometreclimat"
    "RAY_COLOR_PREFIX"        = 1
    "SENTRY_DSN"              = var.sentry_dsn

  }
  # From module specific secrets
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
  # From common secrets
  secret_reference {
    secret_id   = data.scaleway_secret.mediatree_password.id
    environment = "MEDIATREE_PASSWORD"
  }
  secret_reference {
    secret_id   = data.scaleway_secret.openai_api_key.id
    environment = "OPENAI_API_KEY"
  }
  secret_reference {
    secret_id   = data.scaleway_secret.postgres_password_barometre.id
    environment = "POSTGRES_PASSWORD"
  }
  depends_on = [scaleway_container.labelstudio_container]
}
