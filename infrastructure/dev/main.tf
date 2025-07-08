terraform {

  required_providers {
    scaleway = {
      source  = "scaleway/scaleway"
      version = "~> 2.57"
    }
  }
  backend "s3" {
    bucket                      = "climatesafeguards-terraform"
    key                         = "${var.country}/${var.environment}/terraform.tfstate"
    endpoint                    = "https://s3.fr-par.scw.cloud"
    region                      = "fr-par"
    use_lockfile                = true
    skip_credentials_validation = true
    skip_region_validation      = true
    skip_requesting_account_id  = true
  }
}

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

data "scaleway_account_project" "barometre" {
  name = "barometre"
}
data "scaleway_account_project" "project_climatesafeguards" {
  provider = scaleway.project
  name     = "climate-safeguards-${var.environment}"
}

resource "scaleway_iam_application" "project_application" {
  provider    = scaleway.project
  name        = "app-${var.country}-${var.environment}"
  description = "Application handling the permissions for ${var.country} in the ${var.environment} environment"

}

resource "scaleway_iam_policy" "project_application" {
  provider       = scaleway.project
  name           = "app-${var.country}-${var.environment}-policy"
  application_id = scaleway_iam_application.project_application.id
  rule {
    project_ids = [
      data.scaleway_account_project.project_climatesafeguards.id,
      data.scaleway_account_project.barometre.id
    ]
    permission_set_names = [
      "ContainersFullAccess",
      "ContainerRegistryReadOnly",
      "ObjectStorageObjectsRead",
      "ObjectStorageObjectsWrite",
      "ObjectStorageBucketsRead",
      "ObjectStorageBucketsWrite",
      "PrivateNetworksReadOnly",
      "RelationalDatabasesReadOnly",
      "SecretManagerReadOnly",
      "ServerlessJobsFullAccess"
    ]
  }
}

resource "scaleway_iam_api_key" "project_api_key" {
  provider           = scaleway.project
  application_id     = scaleway_iam_application.project_application.id
  description        = "API key for the app-${var.country}-${var.environment} application"
  default_project_id = data.scaleway_account_project.project_climatesafeguards.id
}
