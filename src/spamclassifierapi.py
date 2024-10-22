import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.linear_model import LogisticRegression
import sklearn
import pickle
import boto3
import os
import json
import time
import psycopg2

# Initialize S3 client outside the handler
s3_client = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_ID'],
                         aws_secret_access_key=os.environ['AWS_SECRET_KEY_VAL'])

# load env variables
bucket_name = os.environ['BUCKET_NAME']
artifact_folder = os.environ['MODEL_FOLDER']
model_name_s3 = 'Model.pkl'
vectorizer_name_s3 = 'Vectorizer.pkl'
results_folder = os.environ['RESULTS_FOLDER']


# func to load model from s3
def load_model_from_s3(s3_client, bucket_name, key):
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    body = response['Body'].read()
    return pickle.loads(body)


# Load the model and vectorizer
model_key = os.path.join(artifact_folder, model_name_s3)
vectorizer_key = os.path.join(artifact_folder, vectorizer_name_s3)

# load them to the lambda
model = load_model_from_s3(s3_client, bucket_name, model_key)
vectorizer = load_model_from_s3(s3_client, bucket_name, vectorizer_key)


# a function to detect if a message is spam using the model loaded
def is_spam(message):
    print(f"Input message: {message}")
    inp_series = pd.Series([message])
    inp_transformed = vectorizer.transform(inp_series)
    prediction = model.predict(inp_transformed)
    if prediction == 'spam':
        return True
    else:
        return False


def lambda_handler(event, context):
    body = event.get('body', '')
    data = json.loads(body)
    message = data.get('message', '')
    prediction = is_spam(message)
    response = {
            "message": message,
            "predict": prediction
    }
    
    # write the result back to S3
    timestamp = int(time.time())
    result_filename = f"result_{timestamp}.json"
    result_key = os.path.join(results_folder, result_filename)
    
    s3_client.put_object(
            Bucket=bucket_name,
            Key=result_key,
            Body=json.dumps(response),
            ContentType='application/json'
    )
    
    # output the prediction result
    return {
            'statusCode': 200,
            'body'      : json.dumps(response),
            'headers'   : {'Content-Type': 'application/json'}
    }