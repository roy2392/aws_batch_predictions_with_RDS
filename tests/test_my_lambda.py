import pytest
import json
from unittest.mock import patch, MagicMock
from src.lambda_s3_to_postgres_rds import lambda_handler, is_spam  # Replace 'your_lambda_file' with the actual file name

@pytest.fixture
def mock_env_variables(monkeypatch):
    monkeypatch.setenv('AWS_ACCESS_ID', 'mock_access_id')
    monkeypatch.setenv('AWS_SECRET_KEY_VAL', 'mock_secret_key')
    monkeypatch.setenv('BUCKET_NAME', 'mock_bucket')
    monkeypatch.setenv('MODEL_FOLDER', 'mock_folder/')
    monkeypatch.setenv('DB_HOST', 'mock_host')
    monkeypatch.setenv('DB_NAME', 'mock_db')
    monkeypatch.setenv('DB_USER', 'mock_user')
    monkeypatch.setenv('DB_PASSWORD', 'mock_password')
    monkeypatch.setenv('DB_PORT', '5432')
    monkeypatch.setenv('DB_TABLE', 'mock_table')

@pytest.fixture
def mock_s3():
    with patch('boto3.client') as mock_client:
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3
        yield mock_s3

@pytest.fixture
def mock_psycopg2():
    with patch('psycopg2.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        yield mock_conn

@pytest.fixture
def mock_model():
    with patch('pickle.loads') as mock_loads:
        mock_model = MagicMock()
        mock_loads.return_value = mock_model
        yield mock_model

@pytest.fixture
def mock_vectorizer():
    with patch('pickle.loads') as mock_loads:
        mock_vectorizer = MagicMock()
        mock_loads.return_value = mock_vectorizer
        yield mock_vectorizer

def test_is_spam(mock_model, mock_vectorizer):
    mock_model.predict.return_value = 'spam'
    assert is_spam("Test spam message") == True

    mock_model.predict.return_value = 'ham'
    assert is_spam("Test non-spam message") == False

def test_lambda_handler(mock_env_variables, mock_s3, mock_psycopg2, mock_model, mock_vectorizer):
    event = {
        'body': json.dumps({'text': 'Test message'})
    }
    context = {}

    mock_model.predict.return_value = 'spam'

    response = lambda_handler(event, context)

    assert response['statusCode'] == 200
    assert json.loads(response['body'])['result'] == True

    # Check if database operations were called
    mock_psycopg2.cursor().__enter__().execute.assert_called()
    mock_psycopg2.commit.assert_called_once()

def test_lambda_handler_exception(mock_env_variables, mock_s3, mock_psycopg2, mock_model, mock_vectorizer):
    event = {
        'body': json.dumps({'text': 'Test message'})
    }
    context = {}

    mock_psycopg2.cursor().__enter__().execute.side_effect = Exception("Database error")

    response = lambda_handler(event, context)

    assert response['statusCode'] == 200  # The function still returns 200 even on error
    mock_psycopg2.rollback.assert_called_once()

if __name__ == "__main__":
    pytest.main()