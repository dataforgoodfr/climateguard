## GLOBAL SECRETS
######################################################################

# MEDIATREE_PASSWORD
data "scaleway_secret" "mediatree_password" {
  name       = "mediatree-password"
  path       = "/common"
  project_id = data.scaleway_account_project.project.id
}

# OPENAI_API_KEY
data "scaleway_secret" "openai_api_key" {
  name       = "openai-api-key"
  path       = "/common"
  project_id = data.scaleway_account_project.project.id
}

# POSTGRES_PASSWORD
data "scaleway_secret" "postgres_password_barometre" {
  name       = "postgres_password-barometre"
  path       = "/common"
  project_id = data.scaleway_account_project.project.id
}


## LOCAL SECRETS
######################################################################

# BUCKET_ACCESS
resource "scaleway_secret" "access_key_id" {
  name       = "access-key-${var.country}"
  path       = "/${var.country}"
  project_id = data.scaleway_account_project.project.id
}
resource "scaleway_secret_version" "access_key_id" {
  secret_id = scaleway_secret.access_key_id.id
  data      = scaleway_iam_api_key.project_api_key.access_key
}

# BUCKET_SECRET
resource "scaleway_secret" "secret_key_id" {
  name       = "secret-key-${var.country}"
  path       = "/${var.country}"
  project_id = data.scaleway_account_project.project.id
}
resource "scaleway_secret_version" "secret_key_id" {
  secret_id = scaleway_secret.secret_key_id.id
  data      = scaleway_iam_api_key.project_api_key.secret_key
}

# LABEL_STUDIO_USER_TOKEN
resource "scaleway_secret" "secret_label_studio_token" {
  name       = "labelstudio-${var.country}-token"
  path       = "/${var.country}"
  project_id = data.scaleway_account_project.project.id
}
resource "scaleway_secret_version" "api_label_studio_key_value" {
  secret_id = scaleway_secret.secret_label_studio_token.id
  data      = var.labelstudio_user_token
}

