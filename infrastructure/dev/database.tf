data "scaleway_rdb_instance" "barometre_rdb" {
  name = var.barometre_pg_instance_name
  project_id = var.barometre_project_id
}

# Create PostgreSQL user
resource "scaleway_rdb_user" "labelstudio_user" {
  instance_id = data.scaleway_rdb_instance.barometre_rdb.id
  name        = "labelstudio-${var.subject}-${var.country}-${var.environment}"
  password    = var.postgres_password
}

# Create PostgreSQL database
resource "scaleway_rdb_database" "labelstudio_db" {
  instance_id = data.scaleway_rdb_instance.barometre_rdb.id
  name        = "labelstudio-${var.subject}-${var.country}-${var.environment}"
}

resource "scaleway_rdb_privilege" "labelstudio_policy" {
  instance_id   = data.scaleway_rdb_instance.barometre_rdb.id
  user_name     = scaleway_rdb_user.labelstudio_user.name
  database_name = scaleway_rdb_database.labelstudio_db.name
  permission    = "all"

  depends_on = [scaleway_rdb_user.labelstudio_user, scaleway_rdb_database.labelstudio_db]
}