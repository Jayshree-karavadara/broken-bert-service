"""
Tests for the FastAPI sentiment analysis API.

This module contains tests for the API endpoints.
Note: These tests will skip if the model is not trained yet.
"""

import pytest
import os
import sys
import requests
import time
import subprocess
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))


class TestAPIWithRequests:
    """Test class using requests library (requires running server)."""
    
    @pytest.fixture(scope="class")
    def api_url(self):
        """Base URL for the API."""
        return "http://127.0.0.1:8080"
    
    def test_server_is_running(self, api_url):
        """Test if the server is running (manual test)."""
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            assert response.status_code in [200, 503]
            print(f"✓ Server is running at {api_url}")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running. Start with: uvicorn app.main:app")
    
    def test_root_endpoint(self, api_url):
        """Test the root endpoint."""
        try:
            response = requests.get(f"{api_url}/", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert data["message"] == "Sentiment Analysis API"
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")
    
    def test_health_check(self, api_url):
        """Test the health check endpoint."""
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            assert response.status_code in [200, 503]
            data = response.json()
            assert "status" in data
            assert "model_ready" in data
            assert "message" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")
    
    def test_predict_endpoint_valid_input(self, api_url):
        """Test the predict endpoint with valid input."""
        try:
            test_data = {
                "text": "This movie was absolutely fantastic! Great acting and wonderful storyline."
            }
            response = requests.post(f"{api_url}/predict", json=test_data, timeout=10)
            
            if response.status_code == 503:
                pytest.skip("Model not loaded. Run: python -m ml.train")
            
            assert response.status_code == 200
            data = response.json()
            assert "label" in data
            assert "confidence" in data
            assert data["label"] in ["positive", "negative"]
            assert 0.0 <= data["confidence"] <= 1.0
            # This should be positive based on our fixed label mapping
            assert data["label"] == "positive", f"Expected positive but got {data['label']}"
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")
    
    def test_predict_endpoint_negative_sentiment(self, api_url):
        """Test the predict endpoint with negative sentiment."""
        try:
            test_data = {
                "text": "This movie was terrible and boring, waste of time."
            }
            response = requests.post(f"{api_url}/predict", json=test_data, timeout=10)
            
            if response.status_code == 503:
                pytest.skip("Model not loaded. Run: python -m ml.train")
            
            assert response.status_code == 200
            data = response.json()
            assert "label" in data
            assert "confidence" in data
            # This should be negative based on our fixed label mapping
            assert data["label"] == "negative", f"Expected negative but got {data['label']}"
            assert 0.0 <= data["confidence"] <= 1.0
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")
    
    def test_predict_batch_endpoint(self, api_url):
        """Test the batch predict endpoint."""
        try:
            test_data = {
                "texts": [
                    "Great movie!",
                    "Terrible film.",
                    "Amazing cinematography!"
                ]
            }
            response = requests.post(f"{api_url}/predict/batch", json=test_data, timeout=15)
            
            if response.status_code == 503:
                pytest.skip("Model not loaded. Run: python -m ml.train")
            
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 3
            
            for prediction in data["predictions"]:
                assert "label" in prediction
                assert "confidence" in prediction
                assert prediction["label"] in ["positive", "negative"]
                assert 0.0 <= prediction["confidence"] <= 1.0
            
            # Test specific label mappings
            predictions = data["predictions"]
            assert predictions[0]["label"] == "positive"  # Great movie!
            assert predictions[1]["label"] == "negative"  # Terrible film.
            assert predictions[2]["label"] == "positive"  # Amazing cinematography!
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")
    
    def test_predict_endpoint_invalid_input(self, api_url):
        """Test the predict endpoint with invalid input."""
        try:
            # Empty text
            test_data = {"text": ""}
            response = requests.post(f"{api_url}/predict", json=test_data, timeout=5)
            assert response.status_code == 422
            
            # Missing text field
            test_data = {}
            response = requests.post(f"{api_url}/predict", json=test_data, timeout=5)
            assert response.status_code == 422
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")


class TestAPIOffline:
    """Test class for offline functionality (no server required)."""
    
    def test_app_import(self):
        """Test that we can import the FastAPI app."""
        try:
            from app.main import app
            assert app is not None
            print("✓ FastAPI app imported successfully")
        except Exception as e:
            pytest.fail(f"Failed to import app: {e}")
    
    def test_ml_modules_import(self):
        """Test that ML modules can be imported."""
        try:
            from ml.train import train_model, DistilBertClassifier
            from ml.model import ReviewClassifier
            from ml.data import load_data
            print("✓ All ML modules imported successfully")
        except Exception as e:
            pytest.fail(f"Failed to import ML modules: {e}")
    
    def test_dataset_exists(self):
        """Test that the dataset file exists."""
        dataset_path = Path("assets/reviews.csv")
        assert dataset_path.exists(), f"Dataset not found at {dataset_path}"
        assert dataset_path.stat().st_size > 0, "Dataset file is empty"
        print(f"✓ Dataset found at {dataset_path}")
    
    def test_training_script_help(self):
        """Test that the training script shows help."""
        try:
            result = subprocess.run(
                ["python3", "-m", "ml.train", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0
            assert "Train text classification model" in result.stdout
            print("✓ Training script help works")
        except subprocess.TimeoutExpired:
            pytest.fail("Training script help timed out")
        except Exception as e:
            pytest.fail(f"Training script help failed: {e}")


class TestRecommendationAPI:
    """Test class for product recommendation endpoints."""
    
    @pytest.fixture(scope="class")
    def api_url(self):
        """Base URL for the API."""
        return "http://127.0.0.1:8080"
    
    def test_vector_store_info(self, api_url):
        """Test vector store information endpoint."""
        try:
            response = requests.get(f"{api_url}/vector-store/info", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            print(f"✓ Vector store status: {data['status']}")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")
    
    def test_recommend_endpoint(self, api_url):
        """Test product recommendation endpoint."""
        try:
            test_data = {"text": "Looking for a fast laptop"}
            response = requests.post(f"{api_url}/recommend", json=test_data, timeout=10)
            
            if response.status_code == 503:
                pytest.skip("Qdrant server not available for recommendations")
            
            assert response.status_code == 200
            data = response.json()
            assert "recommended_products" in data
            assert isinstance(data["recommended_products"], list)
            print(f"✓ Recommendations: {data['recommended_products']}")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")
    
    def test_detailed_recommend_endpoint(self, api_url):
        """Test detailed product recommendation endpoint."""
        try:
            test_data = {"text": "wireless headphones"}
            response = requests.post(f"{api_url}/recommend/detailed", json=test_data, timeout=10)
            
            if response.status_code == 503:
                pytest.skip("Qdrant server not available for recommendations")
            
            assert response.status_code == 200
            data = response.json()
            assert "recommendations" in data
            assert isinstance(data["recommendations"], list)
            
            if data["recommendations"]:
                rec = data["recommendations"][0]
                assert "product_id" in rec
                assert "product_title" in rec
                assert "similarity_score" in rec
            
            print(f"✓ Detailed recommendations: {len(data['recommendations'])} items")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")
    
    def test_vector_store_setup(self, api_url):
        """Test vector store setup endpoint."""
        try:
            response = requests.post(f"{api_url}/vector-store/setup", timeout=15)
            
            if response.status_code == 503:
                pytest.skip("Qdrant server not available")
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            print("✓ Vector store setup completed")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema availability."""
        try:
            response = requests.get("http://127.0.0.1:8080/openapi.json", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "openapi" in data
            assert "info" in data
            print("✓ OpenAPI schema available")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")
    
    def test_swagger_ui(self):
        """Test Swagger UI availability."""
        try:
            response = requests.get("http://127.0.0.1:8080/docs", timeout=5)
            assert response.status_code == 200
            assert "swagger" in response.text.lower()
            print("✓ Swagger UI available")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")
    
    def test_redoc(self):
        """Test ReDoc availability."""
        try:
            response = requests.get("http://127.0.0.1:8080/redoc", timeout=5)
            assert response.status_code == 200
            assert "redoc" in response.text.lower()
            print("✓ ReDoc available")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server is not running")


def test_project_structure():
    """Test that the project structure matches expectations."""
    base_path = Path(".")
    
    # Check directories
    assert (base_path / "app").is_dir(), "app/ directory missing"
    assert (base_path / "ml").is_dir(), "ml/ directory missing"
    assert (base_path / "assets").is_dir(), "assets/ directory missing"
    assert (base_path / "tests").is_dir(), "tests/ directory missing"
    
    # Check key files
    assert (base_path / "app" / "main.py").is_file(), "app/main.py missing"
    assert (base_path / "app" / "endpoints.py").is_file(), "app/endpoints.py missing"
    assert (base_path / "app" / "schemas.py").is_file(), "app/schemas.py missing"
    assert (base_path / "ml" / "train.py").is_file(), "ml/train.py missing"
    assert (base_path / "ml" / "model.py").is_file(), "ml/model.py missing"
    assert (base_path / "ml" / "data.py").is_file(), "ml/data.py missing"
    assert (base_path / "assets" / "reviews.csv").is_file(), "assets/reviews.csv missing"
    assert (base_path / "requirements.txt").is_file(), "requirements.txt missing"
    assert (base_path / "README.md").is_file(), "README.md missing"
    
    print("✓ Project structure is correct")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS API TESTS")
    print("="*60)
    print("\nTo run these tests:")
    print("1. Start the server: uvicorn app.main:app --reload")
    print("2. Run tests: pytest tests/test_api.py -v")
    print("\nOr run offline tests only:")
    print("pytest tests/test_api.py::TestAPIOffline -v")
    print("="*60)
    
    # Run tests
    pytest.main([__file__, "-v"])