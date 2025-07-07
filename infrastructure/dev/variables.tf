variable "barometre_pg_instance_name" {
  type    = string
  default = "rdb-poc"
}

variable "postgres_password" {
  type      = string
  sensitive = true
}

variable "country" {
  type    = string
  default = "france"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "subject" {
  type    = string
  default = "climate"
}

variable "labelstudio_admin_password" {
  type = string
  sensitive = true
}

variable "labelstudio_user_token" {
  type = string
  sensitive = true
}