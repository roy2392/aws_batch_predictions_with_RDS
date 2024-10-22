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

aws_access_key_id = os.environ['AWS_ACCESS_ID']
aws_secret_access_key = os.environ['AWS_SECRET_KEY_VAL']

# Initialize S3 client outside the handler
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key)

# load env variables
bucket_name = os.environ['BUCKET_NAME']
artifact_folder = os.environ['MODEL_FOLDER']
model_name_s3 = 'Model.pkl'
vectorizer_name_s3 = 'Vectorizer.pkl'
host = os.environ['DB_HOST']
dbname = os.environ['DB_NAME']
user = os.environ['DB_USER']
password = os.environ['DB_PASSWORD']
port = os.environ['DB_PORT']
output_table = os.environ['DB_TABLE']


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
    try:
        # Establish connection to the Aurora PostgreSQL database
        conn = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port
        )
        
        # Create a cursor object
        cur = conn.cursor()
    
        body = event.get('body', '')
        data = json.loads(body)
        text = data.get('message', '')
        prediction = is_spam(text)
        response = {
                "message": text,
                "predict": prediction
        }
        timestamp = int(time.time())
        
        # Insert the data into the database
        insert_query = """
               INSERT INTO predictions (text, prediction, timestamp)
               VALUES (%s, %s, %s)
               RETURNING id;
               """
        cur.execute(insert_query, (text, prediction, timestamp))
        
        # Fetch the id of the inserted row
        inserted_id = cur.fetchone()[0]
        
        # Commit the transaction
        conn.commit()
        
        # Close the cursor and connection
        cur.close()
        conn.close()
        # Return the result
        return {
                'statusCode': 200,
                'body'      : json.dumps({
                        'message'   : 'Data inserted successfully',
                        'id'        : inserted_id,
                        'text'      : text,
                        'prediction': prediction,
                        'timestamp' : str(timestamp)
                })
        }

    except Exception as e:
        return {
                'statusCode': 500,
                'body'      : json.dumps({'error': str(e)})
        }