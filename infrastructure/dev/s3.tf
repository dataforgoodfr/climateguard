## TEMP
################################################################################

# IAM configuration
data "scaleway_iam_user" "user" {
  email = "giuseppe@dataforgood.fr"
}

################################################################################
resource "scaleway_object_bucket" "source_bucket" {
  name       = "source-${var.country}-${var.environment}"
  project_id = scaleway_account_project.project_climatesafeguards.id
}

resource "scaleway_object_bucket_policy" "source_bucket_policy" {
  bucket = scaleway_object_bucket.source_bucket.name
  policy = jsonencode({
    Version = "2023-04-17",
    Id      = "ApplicationAccessPolicy",
    Statement = [
      {
        Effect    = "Allow"
        Action    = ["s3:*"]
        Principal = { SCW = "application_id:${scaleway_iam_application.project_application.id}" }
        Resource = [
          scaleway_object_bucket.source_bucket.name,
          "${scaleway_object_bucket.source_bucket.name}/*",
        ]
      },
    ## TEMP
    ################################################################################
      {
        Effect    = "Allow"
        Action    = ["s3:*"]
        Principal = { SCW = "user_id:${data.scaleway_iam_user.user.id}" }
        Resource = [
          scaleway_object_bucket.source_bucket.name,
          "${scaleway_object_bucket.source_bucket.name}/*",
        ]
      },
    ################################################################################
    ]
  })
}

