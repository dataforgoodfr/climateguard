## GLOBAL SECRETS
######################################################################

# MEDIATREE_PASSWORD
resource "scaleway_secret" "mediatree_password" {
  name       = "mediatree-password-${var.environment}"
  path       = "global-${var.environment}"
  project_id = scaleway_account_project.project_climatesafeguards.id
}
resource "scaleway_secret_version" "mediatree_password" {
  secret_id = scaleway_secret.mediatree_password.id
  data      = var.mediatree_password
}

# OPENAI_API_KEY
resource "scaleway_secret" "openai_api_key" {
  name       = "openai-api-key-${var.environment}"
  path       = "global-${var.environment}"
  project_id = scaleway_account_project.project_climatesafeguards.id
}
resource "scaleway_secret_version" "openai_api_key" {
  secret_id = scaleway_secret.openai_api_key.id
  data      = var.openai_api_key
}

## LOCAL SECRETS
######################################################################

# BUCKET_ACCESS
resource "scaleway_secret" "access_key_id" {
  name       = "access-key-${var.country}-${var.environment}"
  path       = "${var.country}-${var.environment}"
  project_id = scaleway_account_project.project_climatesafeguards.id
}
resource "scaleway_secret_version" "access_key_id" {
  secret_id = scaleway_secret.access_key_id.id
  data      = scaleway_iam_api_key.project_api_key.access_key
}

# BUCKET_SECRET
resource "scaleway_secret" "secret_key_id" {
  name       = "secret-key-${var.country}-${var.environment}"
  path       = "${var.country}-${var.environment}"
  project_id = scaleway_account_project.project_climatesafeguards.id
}
resource "scaleway_secret_version" "secret_key_id" {
  secret_id = scaleway_secret.secret_key_id.id
  data      = scaleway_iam_api_key.project_api_key.secret_key
}

# LABEL_STUDIO_USER_TOKEN
resource "scaleway_secret" "secret_label_studio_token" {
  name       = "labelstudio-${var.country}-${var.environment}-ls-api"
  project_id = scaleway_account_project.project_climatesafeguards.id
}
resource "scaleway_secret_version" "api_label_studio_key_value" {
  secret_id = scaleway_secret.secret_label_studio_token.id
  data      = var.labelstudio_user_token
}

# POSTGRES_PASSWORD
resource "scaleway_secret" "postgres_password" {
  name       = "postgres_password-key-${var.country}-${var.environment}"
  path       = "${var.country}-${var.environment}"
  project_id = scaleway_account_project.project_climatesafeguards.id
}
resource "scaleway_secret_version" "postgres_password" {
  secret_id = scaleway_secret.postgres_password.id
  data      = var.postgres_password
}