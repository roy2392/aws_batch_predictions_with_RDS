from diagrams import Diagram, Cluster
from diagrams.aws.storage import S3
from diagrams.aws.compute import Lambda
from diagrams.aws.database import RDS
from diagrams.aws.general import User

with Diagram("Spam Detection Pipeline", show=False):
    user = User("User")
    
    with Cluster("AWS Cloud"):
        s3 = S3("Model Storage")
        
        with Cluster("Lambda Function"):
            lambda_func = Lambda("Spam Detection")
        
        rds = RDS("PostgreSQL")
    
    user >> lambda_func
    s3 >> lambda_func
    lambda_func >> rds