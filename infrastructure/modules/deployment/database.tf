data "scaleway_rdb_instance" "barometre_rdb" {
  name       = var.barometre_pg_instance_name
  project_id = data.scaleway_account_project.barometre.id
}

data "scaleway_rdb_database" "barometre" {
  instance_id = data.scaleway_rdb_instance.barometre_rdb.id
  name        = "barometre"
}

# Create PostgreSQL user
resource "scaleway_rdb_user" "labelstudio_user" {
  instance_id = data.scaleway_rdb_instance.barometre_rdb.id
  name        = "labelstudio-${var.subject}-${var.country}-${var.environment}"
  password    = var.postgres_password_labelstudio
}

# Create PostgreSQL database
resource "scaleway_rdb_database" "labelstudio_db" {
  instance_id = data.scaleway_rdb_instance.barometre_rdb.id
  name        = "labelstudio-${var.subject}-${var.country}-${var.environment}-db"
}

resource "scaleway_rdb_privilege" "labelstudio_policy" {
  instance_id   = data.scaleway_rdb_instance.barometre_rdb.id
  user_name     = scaleway_rdb_user.labelstudio_user.name
  database_name = scaleway_rdb_database.labelstudio_db.name
  permission    = "all"

  depends_on = [scaleway_rdb_user.labelstudio_user, scaleway_rdb_database.labelstudio_db]
}


### USING A SINGLE USER FOR THE MOMENT, IN THE FUTURE WE CAN CHANGE THIS TO A MULTI USER 
### BUT TO DO THAT WE NEED TO CREATE AN ADMIN USER IN PROJECTS
### PULL THE USER IN THIS STATE, USER THE POSTGRES PROVIDER TO CONNECT TO THE DB AS ADMIN
### AND THEN ADD CONNECT RIGHTS TO OUR APPLICATION USER WHICH IS A BITCH AND A HALF.


# resource "scaleway_rdb_user" "barometre_user" {
#   instance_id = data.scaleway_rdb_instance.barometre_rdb.id
#   name        = "climateguard-${var.country}-${var.environment}"
#   is_admin    = true
#   password    = var.postgres_password_barometre
# }

# resource "scaleway_rdb_privilege" "barometre_user_policy" {
#   instance_id   = data.scaleway_rdb_instance.barometre_rdb.id
#   user_name     = scaleway_rdb_user.barometre_user.name
#   database_name = data.scaleway_rdb_database.barometre.name
#   permission    = "readonly"

#   depends_on = [scaleway_rdb_user.barometre_user]
# }

# resource "scaleway_rdb_privilege" "labelstudio_barometre_user_policy" {
#   instance_id   = data.scaleway_rdb_instance.barometre_rdb.id
#   user_name     = scaleway_rdb_user.barometre_user.name
#   database_name = scaleway_rdb_database.labelstudio_db.name
#   permission    = "readonly"

#   depends_on = [scaleway_rdb_user.barometre_user, scaleway_rdb_database.labelstudio_db]
# }
