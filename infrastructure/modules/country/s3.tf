## TEMP
################################################################################

# IAM configuration
data "scaleway_iam_user" "user" {
  provider = scaleway.project
  email    = "giuseppe@dataforgood.fr"
}

################################################################################
resource "scaleway_object_bucket" "bucket" {
  provider   = scaleway.project
  name       = "safeguards-${var.subject}-${var.country}-${var.environment}"
  project_id = data.scaleway_account_project.project.id
}

resource "scaleway_object_bucket_policy" "bucket_policy" {
  project_id = data.scaleway_account_project.project.id
  provider   = scaleway.project
  bucket     = scaleway_object_bucket.bucket.name
  policy = jsonencode({
    Version = "2023-04-17",
    Id      = "ApplicationAccessPolicy",
    Statement = [
      {
        Effect    = "Allow"
        Action    = ["s3:*"]
        Principal = { SCW = "application_id:${scaleway_iam_application.project_application.id}" }
        Resource = [
          scaleway_object_bucket.bucket.name,
          "${scaleway_object_bucket.bucket.name}/*",
        ]
      },
      ## TEMP
      ################################################################################
      {
        Effect    = "Allow"
        Action    = ["s3:*"]
        Principal = { SCW = "user_id:${data.scaleway_iam_user.user.id}" }
        Resource = [
          scaleway_object_bucket.bucket.name,
          "${scaleway_object_bucket.bucket.name}/*",
        ]
      },
      ################################################################################
    ]
  })
}

