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

data "scaleway_account_project" "barometre" {
  name = "barometre"
}
resource "scaleway_account_project" "project_climatesafeguards" {
  name = "climatesafeguards-${var.environment}"
}

resource "scaleway_iam_application" "project_application" {
  name        = "app-${var.country}-${var.environment}"
  description = "Application handling the permissions for ${var.country} in the ${var.environment} environment"
}

resource "scaleway_iam_policy" "project_application" {
  name           = "app-${var.country}-${var.environment}-policy"
  application_id = scaleway_iam_application.project_application.id
  rule {
    project_ids = [
      scaleway_account_project.project_climatesafeguards.id,
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
  application_id = scaleway_iam_application.project_application.id
  description = "API key for the app-${var.country}-${var.environment} application"
}
