resource "scaleway_cockpit_grafana_user" "grafana_user" {
  project_id = data.scaleway_account_project.project.id
  login      = "safeguards-${var.environment}-grafana"
  role       = "viewer"
}

output "grafana_password" {
  value = scaleway_cockpit_grafana_user.grafana_user.password
  sensitive = false
}