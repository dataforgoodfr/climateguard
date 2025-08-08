variable "project_secret_access_key" {
  type      = string
  sensitive = true
}

variable "project_access_key_id" {
  type      = string
  sensitive = true
}

variable "project_id" {
  type      = string
  sensitive = true
}

variable "barometre_pg_instance_name" {
  type    = string
  default = "rdb-poc"
}

variable "postgres_password_labelstudio" {
  type      = string
  sensitive = true
}

variable "subject" {
  type    = string
  default = "climate"
}

variable "country" {
  type    = string
  default = "france"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "labelstudio_project" {
  type = number
}

variable "labelstudio_project_id" {
  type = number
}

variable "labelstudio_admin_password" {
  type      = string
  sensitive = true
}

variable "labelstudio_user_token" {
  type      = string
  sensitive = true
}

variable "model_name" {
  type = string
}

variable "sentry_dsn" {
  type        = string
  sensitive = true
}
