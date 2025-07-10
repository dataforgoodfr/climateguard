terraform {

  required_providers {
    scaleway = {
      source  = "scaleway/scaleway"
      version = "~> 2.57"
    }
  }
  backend "s3" {
    bucket                      = "${var.subject}-safeguards-terraform"
    key                         = "${var.environment}/common/terraform.tfstate"
    endpoint                    = "https://s3.fr-par.scw.cloud"
    region                      = "fr-par"
    use_lockfile                = true
    skip_credentials_validation = true
    skip_region_validation      = true
    skip_requesting_account_id  = true
  }
}


module "common" {
  source                      = "../../../modules/common/"
  mediatree_password          = var.mediatree_password
  postgres_password_barometre = var.postgres_password_barometre
  openai_api_key              = var.openai_api_key
  subject                     = var.subject
  environment                 = var.environment
  project_id                  = var.project_id
  project_access_key_id       = var.project_access_key_id
  project_secret_access_key   = var.project_secret_access_key
}