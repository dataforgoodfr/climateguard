resource "scaleway_account_project" "project_climatesafeguards_dev" {
  name = "climate-safeguards-dev"
}

resource "scaleway_iam_application" "climate_safeguards_dev_tofu" {
  name        = "climate-safeguards-dev-tofu"
  description = "IAM Application handling the credentials to deploy resources in the climate-safeguards-dev project"
}