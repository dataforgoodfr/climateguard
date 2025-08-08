resource "scaleway_account_project" "project_climatesafeguards_prod" {
  name = "climate-safeguards-prod"
}

resource "scaleway_iam_application" "climate_safeguards_prod_tofu" {
  name        = "climate-safeguards-prod-tofu"
  description = "IAM Application handling the credentials to deploy resources in the climate-safeguards-prod project"
}