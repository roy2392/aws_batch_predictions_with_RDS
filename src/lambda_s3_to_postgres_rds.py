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
from sqlalchemy import create_engine

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
    timestemp = datetime.date.today().strftime("%B %d, %Y")

    print("trying to write")
    new_row = pd.DataFrame([[text_value, result, timestemp]],
                           columns=['text', 'prediction_result', 'timestamp'])

    # Create SQLAlchemy engine
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')

    # Insert the DataFrame into the PostgreSQL table
    new_row.to_sql(output_table, engine, if_exists='append', index=False)
    print("success")

    # Query the table to verify insertion
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
        print(df)

    return {
            'statusCode': 200,
            'body'      : json.dumps({'result': result})
    }

if __name__ == "__main__":
    import json

    # Load test event
    with open('src/tests/test_event.json', 'r') as f:
        test_event = json.load(f)

    # Simulate context (can be an empty object for testing)
    test_context = {}

    # Call the lambda_handler function
    result = lambda_handler(test_event, test_context)
    print(result)