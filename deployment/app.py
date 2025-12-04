#!/usr/bin/env python3
"""
AWS CDK Infrastructure for EV Battery Prediction System
Deploys S3, SageMaker, ECS, and monitoring resources.
"""

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_elasticloadbalancingv2 as elbv2,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_lambda as lambda_,
    aws_events as events,
    aws_events_targets as targets,
    aws_kms as kms,
    Duration,
    RemovalPolicy
)
from constructs import Construct

class EVBatteryStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # KMS Key for encryption
        self.kms_key = kms.Key(
            self, "EVBatteryKey",
            description="KMS key for EV Battery prediction system",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.DESTROY
        )
        
        # S3 Buckets
        self.create_s3_buckets()
        
        # IAM Roles
        self.create_iam_roles()
        
        # VPC and Networking
        self.create_vpc()
        
        # SageMaker Resources
        self.create_sagemaker_resources()
        
        # ECS Cluster and Service
        self.create_ecs_resources()
        
        # Monitoring and Alerting
        self.create_monitoring()
        
        # Lambda Functions
        self.create_lambda_functions()
    
    def create_s3_buckets(self):
        """Create S3 buckets for data storage"""
        # Raw data bucket
        self.raw_data_bucket = s3.Bucket(
            self, "RawDataBucket",
            bucket_name=f"ev-battery-raw-data-{self.account}-{self.region}",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.kms_key,
            versioned=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ArchiveOldData",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30)
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90)
                        )
                    ]
                )
            ],
            removal_policy=RemovalPolicy.DESTROY
        )
        
        # Processed data bucket
        self.processed_data_bucket = s3.Bucket(
            self, "ProcessedDataBucket",
            bucket_name=f"ev-battery-processed-data-{self.account}-{self.region}",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.kms_key,
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY
        )
        
        # Model artifacts bucket
        self.model_bucket = s3.Bucket(
            self, "ModelBucket",
            bucket_name=f"ev-battery-models-{self.account}-{self.region}",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.kms_key,
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY
        )
        
        # Dashboard assets bucket
        self.dashboard_bucket = s3.Bucket(
            self, "DashboardBucket",
            bucket_name=f"ev-battery-dashboard-{self.account}-{self.region}",
            website_index_document="index.html",
            public_read_access=True,
            removal_policy=RemovalPolicy.DESTROY
        )
    
    def create_iam_roles(self):
        """Create IAM roles for various services"""
        # SageMaker execution role
        self.sagemaker_role = iam.Role(
            self, "SageMakerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
            ]
        )
        
        # Grant S3 access to SageMaker role
        for bucket in [self.raw_data_bucket, self.processed_data_bucket, self.model_bucket]:
            bucket.grant_read_write(self.sagemaker_role)
        
        # ECS task role
        self.ecs_task_role = iam.Role(
            self, "ECSTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            inline_policies={
                "S3Access": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=["s3:GetObject", "s3:PutObject"],
                            resources=[
                                f"{self.processed_data_bucket.bucket_arn}/*",
                                f"{self.model_bucket.bucket_arn}/*"
                            ]
                        )
                    ]
                )
            }
        )
        
        # ECS execution role
        self.ecs_execution_role = iam.Role(
            self, "ECSExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
            ]
        )
    
    def create_vpc(self):
        """Create VPC and networking resources"""
        self.vpc = ec2.Vpc(
            self, "EVBatteryVPC",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ]
        )
        
        # Security group for ECS
        self.ecs_security_group = ec2.SecurityGroup(
            self, "ECSSecurityGroup",
            vpc=self.vpc,
            description="Security group for ECS tasks",
            allow_all_outbound=True
        )
        
        self.ecs_security_group.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(8000),
            description="Allow HTTP traffic"
        )
    
    def create_sagemaker_resources(self):
        """Create SageMaker resources"""
        # SageMaker notebook instance
        self.notebook_instance = sagemaker.CfnNotebookInstance(
            self, "EVBatteryNotebook",
            instance_type="ml.t3.medium",
            role_arn=self.sagemaker_role.role_arn,
            notebook_instance_name="ev-battery-notebook",
            default_code_repository="https://github.com/your-repo/ev-battery-prediction.git",
            volume_size_in_gb=20
        )
        
        # SageMaker model (placeholder - will be created by training job)
        self.model_name = "ev-battery-baseline-model"
    
    def create_ecs_resources(self):
        """Create ECS cluster and service"""
        # ECS Cluster
        self.ecs_cluster = ecs.Cluster(
            self, "EVBatteryCluster",
            vpc=self.vpc,
            cluster_name="ev-battery-cluster"
        )
        
        # Task definition
        self.task_definition = ecs.FargateTaskDefinition(
            self, "EVBatteryTaskDef",
            memory_limit_mib=2048,
            cpu=1024,
            task_role=self.ecs_task_role,
            execution_role=self.ecs_execution_role
        )
        
        # Container definition
        self.container = self.task_definition.add_container(
            "EVBatteryAPI",
            image=ecs.ContainerImage.from_registry("python:3.9-slim"),
            memory_limit_mib=2048,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="ev-battery-api",
                log_retention=logs.RetentionDays.ONE_WEEK
            ),
            environment={
                "MODEL_BUCKET": self.model_bucket.bucket_name,
                "DATA_BUCKET": self.processed_data_bucket.bucket_name
            }
        )
        
        self.container.add_port_mappings(
            ecs.PortMapping(container_port=8000, protocol=ecs.Protocol.TCP)
        )
        
        # ECS Service
        self.ecs_service = ecs.FargateService(
            self, "EVBatteryService",
            cluster=self.ecs_cluster,
            task_definition=self.task_definition,
            desired_count=2,
            security_groups=[self.ecs_security_group],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)
        )
        
        # Application Load Balancer
        self.alb = elbv2.ApplicationLoadBalancer(
            self, "EVBatteryALB",
            vpc=self.vpc,
            internet_facing=True,
            security_group=self.ecs_security_group
        )
        
        # Target group
        self.target_group = elbv2.ApplicationTargetGroup(
            self, "EVBatteryTargets",
            port=8000,
            vpc=self.vpc,
            target_type=elbv2.TargetType.IP,
            health_check=elbv2.HealthCheck(
                path="/health",
                healthy_http_codes="200"
            )
        )
        
        # Listener
        self.alb.add_listener(
            "EVBatteryListener",
            port=80,
            default_target_groups=[self.target_group]
        )
        
        # Connect service to target group
        self.ecs_service.attach_to_application_target_group(self.target_group)
    
    def create_monitoring(self):
        """Create CloudWatch dashboards and alarms"""
        # CloudWatch Dashboard
        self.dashboard = cloudwatch.Dashboard(
            self, "EVBatteryDashboard",
            dashboard_name="EV-Battery-Monitoring"
        )
        
        # Add widgets to dashboard
        self.dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="API Response Time",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/ApplicationELB",
                        metric_name="TargetResponseTime",
                        dimensions_map={
                            "LoadBalancer": self.alb.load_balancer_full_name
                        }
                    )
                ]
            ),
            cloudwatch.GraphWidget(
                title="API Request Count",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/ApplicationELB",
                        metric_name="RequestCount",
                        dimensions_map={
                            "LoadBalancer": self.alb.load_balancer_full_name
                        }
                    )
                ]
            )
        )
        
        # Alarms
        cloudwatch.Alarm(
            self, "HighResponseTimeAlarm",
            metric=cloudwatch.Metric(
                namespace="AWS/ApplicationELB",
                metric_name="TargetResponseTime",
                dimensions_map={
                    "LoadBalancer": self.alb.load_balancer_full_name
                }
            ),
            threshold=5.0,
            evaluation_periods=2,
            alarm_description="API response time is too high"
        )
        
        cloudwatch.Alarm(
            self, "HighErrorRateAlarm",
            metric=cloudwatch.Metric(
                namespace="AWS/ApplicationELB",
                metric_name="HTTPCode_Target_5XX_Count",
                dimensions_map={
                    "LoadBalancer": self.alb.load_balancer_full_name
                }
            ),
            threshold=10,
            evaluation_periods=2,
            alarm_description="High error rate detected"
        )
    
    def create_lambda_functions(self):
        """Create Lambda functions for data processing and monitoring"""
        # Data drift detection function
        self.drift_detection_function = lambda_.Function(
            self, "DriftDetectionFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="drift_detection.handler",
            code=lambda_.Code.from_asset("lambda"),
            timeout=Duration.minutes(5),
            memory_size=512,
            environment={
                "DATA_BUCKET": self.processed_data_bucket.bucket_name,
                "MODEL_BUCKET": self.model_bucket.bucket_name
            }
        )
        
        # Grant permissions
        self.processed_data_bucket.grant_read(self.drift_detection_function)
        self.model_bucket.grant_read(self.drift_detection_function)
        
        # Schedule drift detection
        events.Rule(
            self, "DriftDetectionSchedule",
            schedule=events.Schedule.rate(Duration.hours(24)),
            targets=[targets.LambdaFunction(self.drift_detection_function)]
        )
        
        # Model retraining trigger function
        self.retrain_function = lambda_.Function(
            self, "RetrainFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="retrain_trigger.handler",
            code=lambda_.Code.from_asset("lambda"),
            timeout=Duration.minutes(1),
            environment={
                "SAGEMAKER_ROLE": self.sagemaker_role.role_arn,
                "MODEL_BUCKET": self.model_bucket.bucket_name
            }
        )
        
        # Grant SageMaker permissions
        self.retrain_function.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sagemaker:CreateTrainingJob"],
                resources=["*"]
            )
        )

# CDK App
app = cdk.App()
EVBatteryStack(app, "EVBatteryStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1"
    )
)

app.synth()