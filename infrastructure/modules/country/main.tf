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

data "scaleway_account_project" "project" {
  provider = scaleway.project
  name     = "${var.subject}-safeguards-${var.environment}"
}

resource "scaleway_iam_application" "project_application" {
  provider    = scaleway.project
  name        = "app-${var.subject}-${var.country}-${var.environment}"
  description = "Application handling the permissions for the label misinformation for project ${data.scaleway_account_project.project.name}"
}

resource "scaleway_iam_policy" "project_application" {
  provider       = scaleway.project
  name           = "app-${var.subject}-${var.country}-${var.environment}-policy"
  application_id = scaleway_iam_application.project_application.id
  rule {
    project_ids = [
      data.scaleway_account_project.project.id,
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
  description        = "API key for the app-${var.subject}-${var.country}-${var.environment} application"
  default_project_id = data.scaleway_account_project.project.id
}
