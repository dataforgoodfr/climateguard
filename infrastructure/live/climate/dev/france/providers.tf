
provider "scaleway" {
  alias  = "state"
  region = "fr-par"
}

provider "scaleway" {
  alias      = "project"
  region     = "fr-par"
  project_id = var.project_id
  access_key = var.project_access_key_id
  secret_key = var.project_secret_access_key
}