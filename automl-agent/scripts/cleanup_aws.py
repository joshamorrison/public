#!/usr/bin/env python3
"""
AWS Cleanup Script for AutoML Agent Platform

Safely removes AWS resources created by the deployment script.
Use with caution - this will delete resources and data!
"""

import boto3
import argparse
import sys
from datetime import datetime
from typing import List, Dict, Any

class AutoMLAWSCleanup:
    """Cleanup AWS resources for AutoML Platform"""
    
    def __init__(self, region_name: str = 'us-east-1', environment: str = 'production'):
        self.region_name = region_name
        self.environment = environment
        self.session = boto3.Session(region_name=region_name)
        
        # AWS service clients
        self.s3 = self.session.client('s3')
        self.ecs = self.session.client('ecs')
        self.lambda_client = self.session.client('lambda')
        self.sagemaker = self.session.client('sagemaker')
        self.cloudformation = self.session.client('cloudformation')
        self.ecr = self.session.client('ecr')
        self.cloudwatch = self.session.client('logs')
        
        self.resources_to_cleanup = {
            'cloudformation_stacks': [],
            's3_buckets': [],
            'ecs_services': [],
            'lambda_functions': [],
            'sagemaker_endpoints': [],
            'ecr_repositories': [],
            'log_groups': []
        }
    
    def find_automl_resources(self) -> Dict[str, List[str]]:
        """Find all AutoML-related resources"""
        print("🔍 Scanning for AutoML resources...")
        
        # Find CloudFormation stacks
        try:
            stacks = self.cloudformation.list_stacks(
                StackStatusFilter=['CREATE_COMPLETE', 'UPDATE_COMPLETE']
            )
            for stack in stacks['StackSummaries']:
                if 'automl' in stack['StackName'].lower():
                    self.resources_to_cleanup['cloudformation_stacks'].append(stack['StackName'])
        except Exception as e:
            print(f"⚠️  Could not scan CloudFormation: {e}")
        
        # Find S3 buckets
        try:
            buckets = self.s3.list_buckets()
            for bucket in buckets['Buckets']:
                if 'automl' in bucket['Name'].lower():
                    self.resources_to_cleanup['s3_buckets'].append(bucket['Name'])
        except Exception as e:
            print(f"⚠️  Could not scan S3: {e}")
        
        # Find ECS services and clusters
        try:
            clusters = self.ecs.list_clusters()
            for cluster_arn in clusters['clusterArns']:
                cluster_name = cluster_arn.split('/')[-1]
                if 'automl' in cluster_name.lower():
                    services = self.ecs.list_services(cluster=cluster_name)
                    for service_arn in services['serviceArns']:
                        self.resources_to_cleanup['ecs_services'].append({
                            'cluster': cluster_name,
                            'service': service_arn.split('/')[-1]
                        })
        except Exception as e:
            print(f"⚠️  Could not scan ECS: {e}")
        
        # Find Lambda functions
        try:
            functions = self.lambda_client.list_functions()
            for function in functions['Functions']:
                if 'automl' in function['FunctionName'].lower():
                    self.resources_to_cleanup['lambda_functions'].append(function['FunctionName'])
        except Exception as e:
            print(f"⚠️  Could not scan Lambda: {e}")
        
        # Find SageMaker endpoints
        try:
            endpoints = self.sagemaker.list_endpoints()
            for endpoint in endpoints['Endpoints']:
                if 'automl' in endpoint['EndpointName'].lower():
                    self.resources_to_cleanup['sagemaker_endpoints'].append(endpoint['EndpointName'])
        except Exception as e:
            print(f"⚠️  Could not scan SageMaker: {e}")
        
        # Find ECR repositories
        try:
            repositories = self.ecr.describe_repositories()
            for repo in repositories['repositories']:
                if 'automl' in repo['repositoryName'].lower():
                    self.resources_to_cleanup['ecr_repositories'].append(repo['repositoryName'])
        except Exception as e:
            print(f"⚠️  Could not scan ECR: {e}")
        
        # Find CloudWatch log groups
        try:
            log_groups = self.cloudwatch.describe_log_groups()
            for log_group in log_groups['logGroups']:
                if 'automl' in log_group['logGroupName'].lower():
                    self.resources_to_cleanup['log_groups'].append(log_group['logGroupName'])
        except Exception as e:
            print(f"⚠️  Could not scan CloudWatch: {e}")
        
        return self.resources_to_cleanup
    
    def print_cleanup_plan(self):
        """Display what will be deleted"""
        print("\n" + "=" * 60)
        print("🗑️  CLEANUP PLAN - The following resources will be DELETED:")
        print("=" * 60)
        
        total_resources = 0
        for resource_type, resources in self.resources_to_cleanup.items():
            if resources:
                print(f"\n📦 {resource_type.upper().replace('_', ' ')}:")
                for resource in resources:
                    if isinstance(resource, dict):
                        print(f"   - {resource}")
                    else:
                        print(f"   - {resource}")
                total_resources += len(resources)
        
        if total_resources == 0:
            print("\n✅ No AutoML resources found to cleanup")
            return False
        
        print(f"\n📊 Total resources to delete: {total_resources}")
        print("\n⚠️  WARNING: This action cannot be undone!")
        print("⚠️  All data in S3 buckets will be permanently lost!")
        return True
    
    def cleanup_cloudformation_stacks(self) -> int:
        """Delete CloudFormation stacks"""
        deleted = 0
        for stack_name in self.resources_to_cleanup['cloudformation_stacks']:
            try:
                print(f"🗑️  Deleting CloudFormation stack: {stack_name}")
                self.cloudformation.delete_stack(StackName=stack_name)
                deleted += 1
            except Exception as e:
                print(f"❌ Failed to delete stack {stack_name}: {e}")
        return deleted
    
    def cleanup_s3_buckets(self) -> int:
        """Delete S3 buckets and all contents"""
        deleted = 0
        for bucket_name in self.resources_to_cleanup['s3_buckets']:
            try:
                print(f"🗑️  Emptying S3 bucket: {bucket_name}")
                
                # Delete all objects in bucket
                paginator = self.s3.get_paginator('list_object_versions')
                for page in paginator.paginate(Bucket=bucket_name):
                    if 'Versions' in page:
                        delete_keys = [{'Key': obj['Key'], 'VersionId': obj['VersionId']} 
                                     for obj in page['Versions']]
                        if delete_keys:
                            self.s3.delete_objects(
                                Bucket=bucket_name,
                                Delete={'Objects': delete_keys}
                            )
                    
                    if 'DeleteMarkers' in page:
                        delete_keys = [{'Key': obj['Key'], 'VersionId': obj['VersionId']} 
                                     for obj in page['DeleteMarkers']]
                        if delete_keys:
                            self.s3.delete_objects(
                                Bucket=bucket_name,
                                Delete={'Objects': delete_keys}
                            )
                
                # Delete the bucket
                print(f"🗑️  Deleting S3 bucket: {bucket_name}")
                self.s3.delete_bucket(Bucket=bucket_name)
                deleted += 1
                
            except Exception as e:
                print(f"❌ Failed to delete bucket {bucket_name}: {e}")
        return deleted
    
    def cleanup_ecs_services(self) -> int:
        """Delete ECS services and clusters"""
        deleted = 0
        clusters_to_delete = set()
        
        for service_info in self.resources_to_cleanup['ecs_services']:
            try:
                cluster = service_info['cluster']
                service = service_info['service']
                
                print(f"🗑️  Scaling down ECS service: {service}")
                # Scale service to 0
                self.ecs.update_service(
                    cluster=cluster,
                    service=service,
                    desiredCount=0
                )
                
                # Delete service
                print(f"🗑️  Deleting ECS service: {service}")
                self.ecs.delete_service(
                    cluster=cluster,
                    service=service,
                    force=True
                )
                
                clusters_to_delete.add(cluster)
                deleted += 1
                
            except Exception as e:
                print(f"❌ Failed to delete service {service}: {e}")
        
        # Delete empty clusters
        for cluster in clusters_to_delete:
            try:
                print(f"🗑️  Deleting ECS cluster: {cluster}")
                self.ecs.delete_cluster(cluster=cluster)
            except Exception as e:
                print(f"❌ Failed to delete cluster {cluster}: {e}")
        
        return deleted
    
    def cleanup_lambda_functions(self) -> int:
        """Delete Lambda functions"""
        deleted = 0
        for function_name in self.resources_to_cleanup['lambda_functions']:
            try:
                print(f"🗑️  Deleting Lambda function: {function_name}")
                self.lambda_client.delete_function(FunctionName=function_name)
                deleted += 1
            except Exception as e:
                print(f"❌ Failed to delete function {function_name}: {e}")
        return deleted
    
    def cleanup_sagemaker_endpoints(self) -> int:
        """Delete SageMaker endpoints"""
        deleted = 0
        for endpoint_name in self.resources_to_cleanup['sagemaker_endpoints']:
            try:
                print(f"🗑️  Deleting SageMaker endpoint: {endpoint_name}")
                self.sagemaker.delete_endpoint(EndpointName=endpoint_name)
                deleted += 1
            except Exception as e:
                print(f"❌ Failed to delete endpoint {endpoint_name}: {e}")
        return deleted
    
    def cleanup_ecr_repositories(self) -> int:
        """Delete ECR repositories"""
        deleted = 0
        for repo_name in self.resources_to_cleanup['ecr_repositories']:
            try:
                print(f"🗑️  Deleting ECR repository: {repo_name}")
                self.ecr.delete_repository(repositoryName=repo_name, force=True)
                deleted += 1
            except Exception as e:
                print(f"❌ Failed to delete repository {repo_name}: {e}")
        return deleted
    
    def cleanup_log_groups(self) -> int:
        """Delete CloudWatch log groups"""
        deleted = 0
        for log_group in self.resources_to_cleanup['log_groups']:
            try:
                print(f"🗑️  Deleting CloudWatch log group: {log_group}")
                self.cloudwatch.delete_log_group(logGroupName=log_group)
                deleted += 1
            except Exception as e:
                print(f"❌ Failed to delete log group {log_group}: {e}")
        return deleted
    
    def run_cleanup(self, confirm: bool = False) -> bool:
        """Run the complete cleanup process"""
        # Scan for resources
        self.find_automl_resources()
        
        # Show cleanup plan
        has_resources = self.print_cleanup_plan()
        if not has_resources:
            return True
        
        # Get confirmation
        if not confirm:
            response = input("\n❓ Are you sure you want to delete these resources? (type 'DELETE' to confirm): ")
            if response != 'DELETE':
                print("🚫 Cleanup cancelled")
                return False
        
        print(f"\n🗑️  Starting cleanup process...")
        print("=" * 60)
        
        # Cleanup in order (reverse of deployment)
        cleanup_functions = [
            ("CloudFormation Stacks", self.cleanup_cloudformation_stacks),
            ("ECS Services", self.cleanup_ecs_services),
            ("Lambda Functions", self.cleanup_lambda_functions),
            ("SageMaker Endpoints", self.cleanup_sagemaker_endpoints),
            ("ECR Repositories", self.cleanup_ecr_repositories),
            ("CloudWatch Log Groups", self.cleanup_log_groups),
            ("S3 Buckets", self.cleanup_s3_buckets)  # S3 last to avoid dependency issues
        ]
        
        total_deleted = 0
        for description, cleanup_func in cleanup_functions:
            print(f"\n🧹 Cleaning up {description}...")
            deleted = cleanup_func()
            total_deleted += deleted
            if deleted > 0:
                print(f"✅ Deleted {deleted} {description.lower()}")
        
        print(f"\n" + "=" * 60)
        print(f"🎯 Cleanup Summary:")
        print(f"   Total resources deleted: {total_deleted}")
        print(f"   Environment: {self.environment}")
        print(f"   Region: {self.region_name}")
        print(f"✅ AutoML Platform cleanup completed!")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Cleanup AutoML Platform AWS resources')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--environment', default='production', choices=['dev', 'staging', 'production'])
    parser.add_argument('--confirm', action='store_true', help='Skip confirmation prompt')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without doing it')
    
    args = parser.parse_args()
    
    # Initialize cleanup
    cleanup = AutoMLAWSCleanup(
        region_name=args.region,
        environment=args.environment
    )
    
    if args.dry_run:
        print("🔍 Dry run mode - showing resources that would be deleted...")
        cleanup.find_automl_resources()
        cleanup.print_cleanup_plan()
    else:
        success = cleanup.run_cleanup(args.confirm)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()