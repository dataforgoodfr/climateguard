variable "subject" {
  type    = string
  default = "climate"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "postgres_password_barometre" {
  type      = string
  sensitive = true
}

variable "mediatree_password" {
  type      = string
  sensitive = true
}

variable "openai_api_key" {
  type      = string
  sensitive = true
}

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