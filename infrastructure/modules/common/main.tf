provider "scaleway" {
  alias      = "project"
  region     = "fr-par"
  project_id = var.project_id
  access_key = var.project_access_key_id
  secret_key = var.project_secret_access_key
}

data "scaleway_account_project" "project" {
  provider = scaleway.project
  name     = "${var.subject}-safeguards-${var.environment}"
}

resource "scaleway_container_namespace" "container_namespace" {
  name       = "${var.subject}-safeguards-containers-${var.environment}"
  project_id = data.scaleway_account_project.project.id
}