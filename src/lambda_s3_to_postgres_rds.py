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
import datetime

# Initialize aws credentials
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_KEY_VAL']

# Initialize S3 client outside the handler
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# Load env variables
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
query = f"SELECT * FROM {output_table}"

# Load model pkl
response = s3.get_object(Bucket=bucket_name, Key=f'{artifact_folder}{model_name_s3}')
body = response['Body'].read()
model = pickle.loads(body)

# Load vectorizer pkl
response = s3.get_object(Bucket=bucket_name, Key=f'{artifact_folder}{vectorizer_name_s3}')
body = response['Body'].read()
vectorizer = pickle.loads(body)


def is_spam(inp):
    print(inp)
    inp = pd.Series(inp)
    inp_test = vectorizer.transform(inp)
    inp_sonuc = model.predict(inp_test)
    
    
    if inp_sonuc == 'spam':
        return True
    else:
        return False


def lambda_handler(event, context):
    body_json = json.loads(event['body'])
    text_value = body_json['text']
    result = is_spam(text_value)
    timestamp = datetime.date.today().strftime("%Y-%m-%d")
    
    print("trying to write")
    
    # Connect to the database
    conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
    )
    
    try:
        with conn.cursor() as cur:
            # Create table if it doesn't exist
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {output_table} (
                    id SERIAL PRIMARY KEY,
                    text TEXT,
                    prediction_result BOOLEAN,
                    timestamp DATE
                )
            """)
            
            # Insert new row
            cur.execute(f"""
                INSERT INTO {output_table} (text, prediction_result, timestamp)
                VALUES (%s, %s, %s)
            """, (text_value, result, timestamp))
            
            conn.commit()
            
            print("success")
            
            # Fetch and print all rows
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                print(row)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()
    
    return {
            'statusCode': 200,
            'body'      : json.dumps({'result': result})
    }