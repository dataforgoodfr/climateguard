module "france" {
  source = "../../../../modules/country"
  # variables
  subject                    = var.subject
  environment                = var.environment
  country                    = var.country
  job_cron_schedule          = "5 6 * * *"
  job_cron_schedule_timezone = "Europe/Paris"
  # Defined as secrets
  project_secret_access_key     = var.project_secret_access_key
  project_access_key_id         = var.project_access_key_id
  project_id                    = var.project_id
  barometre_pg_instance_name    = var.barometre_pg_instance_name
  postgres_password_labelstudio = var.postgres_password_labelstudio
  labelstudio_project           = var.labelstudio_project
  labelstudio_project_id        = var.labelstudio_project_id
  labelstudio_admin_password    = var.labelstudio_admin_password
  labelstudio_user_token        = var.labelstudio_user_token
  model_name                    = var.model_name
  sentry_dsn                    = var.sentry_dsn
}
