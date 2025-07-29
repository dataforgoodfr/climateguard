


module "common" {
  source                      = "../../../../modules/common/"
  subject                     = var.subject
  environment                 = var.environment
  # Secrets from .env.secrets
  mediatree_password          = var.mediatree_password
  postgres_password_barometre = var.postgres_password_barometre
  openai_api_key              = var.openai_api_key
  project_id                  = var.project_id
  project_access_key_id       = var.project_access_key_id
  project_secret_access_key   = var.project_secret_access_key
}