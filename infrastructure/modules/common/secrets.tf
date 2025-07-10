
# MEDIATREE_PASSWORD
resource "scaleway_secret" "mediatree_password" {
  name       = "mediatree-password"
  path       = "/global"
  project_id = data.scaleway_account_project.project.id
}
resource "scaleway_secret_version" "mediatree_password" {
  secret_id = scaleway_secret.mediatree_password.id
  data      = var.mediatree_password
}

# OPENAI_API_KEY
resource "scaleway_secret" "openai_api_key" {
  name       = "openai-api-key"
  path       = "/global"
  project_id = data.scaleway_account_project.project.id
}
resource "scaleway_secret_version" "openai_api_key" {
  secret_id = scaleway_secret.openai_api_key.id
  data      = var.openai_api_key
}

# POSTGRES_PASSWORD
resource "scaleway_secret" "postgres_password_barometre" {
  name       = "postgres_password-barometre"
  path       = "/common"
  project_id = data.scaleway_account_project.project.id
}
resource "scaleway_secret_version" "postgres_password_barometre" {
  secret_id = scaleway_secret.postgres_password_barometre.id
  data      = var.postgres_password_barometre
}