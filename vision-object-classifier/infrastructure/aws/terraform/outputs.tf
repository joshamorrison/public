# Terraform outputs

output "alb_hostname" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.main.zone_id
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.main.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.app.name
}

output "s3_models_bucket" {
  description = "S3 bucket for models"
  value       = aws_s3_bucket.models.bucket
}

output "s3_outputs_bucket" {
  description = "S3 bucket for outputs"
  value       = aws_s3_bucket.outputs.bucket
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "List of private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of public subnet IDs"
  value       = module.vpc.public_subnets
}

output "security_group_alb" {
  description = "Security group ID for ALB"
  value       = aws_security_group.alb.id
}

output "security_group_ecs_tasks" {
  description = "Security group ID for ECS tasks"
  value       = aws_security_group.ecs_tasks.id
}

output "api_url" {
  description = "URL to access the API"
  value       = "http://${aws_lb.main.dns_name}"
}

output "health_check_url" {
  description = "Health check endpoint"
  value       = "http://${aws_lb.main.dns_name}/health/status"
}

output "api_docs_url" {
  description = "API documentation URL"
  value       = "http://${aws_lb.main.dns_name}/docs"
}