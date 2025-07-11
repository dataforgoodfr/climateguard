terraform {

  required_providers {
    scaleway = {
      source  = "scaleway/scaleway"
      version = "~> 2.57"
    }
  }
  backend "s3" {
    bucket                      = "climatesafeguards-terraform"
    key                         = "common/terraform.tfstate"
    endpoint                    = "https://s3.fr-par.scw.cloud"
    region                      = "fr-par"
    use_lockfile                = true
    skip_credentials_validation = true
    skip_region_validation      = true
    skip_requesting_account_id  = true

  }
}

provider "scaleway" {
  region = "fr-par"
}

