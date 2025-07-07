data "scaleway_rdb_instance" "barometre_rdb" {
  name = var.barometre_pg_instance_name
  project_id = data.scaleway_account_project.barometre.id
}

data "scaleway_rdb_database" "barometre" {
  instance_id = data.scaleway_rdb_instance.barometre_rdb.id
  name = "barometre"
}

resource "scaleway_rdb_user" "barometre_user" {
  instance_id = data.scaleway_rdb_instance.barometre_rdb.id
  name        = "barometre-${var.country}-${var.environment}"
  password    = var.postgres_password_barometre
}

resource "scaleway_rdb_privilege" "barometre_user_policy" {
  instance_id   = data.scaleway_rdb_instance.barometre_rdb.id
  user_name     = scaleway_rdb_user.barometre_user.name
  database_name = data.scaleway_rdb_database.barometre.name
  permission    = "readonly"

  depends_on = [ scaleway_rdb_user.barometre_user ]
}

# Create PostgreSQL user
resource "scaleway_rdb_user" "labelstudio_user" {
  instance_id = data.scaleway_rdb_instance.barometre_rdb.id
  name        = "labelstudio-${var.country}-${var.environment}"
  password    = var.postgres_password_labelstudio
}

# Create PostgreSQL database
resource "scaleway_rdb_database" "labelstudio_db" {
  instance_id = data.scaleway_rdb_instance.barometre_rdb.id
  name        = "labelstudio-${var.country}-${var.environment}"
}

resource "scaleway_rdb_privilege" "labelstudio_policy" {
  instance_id   = data.scaleway_rdb_instance.barometre_rdb.id
  user_name     = scaleway_rdb_user.labelstudio_user.name
  database_name = scaleway_rdb_database.labelstudio_db.name
  permission    = "all"

  depends_on = [scaleway_rdb_user.labelstudio_user, scaleway_rdb_database.labelstudio_db]
}