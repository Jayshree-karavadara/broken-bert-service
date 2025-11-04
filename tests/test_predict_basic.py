"""
Basic Unit Tests for /predict Endpoint

This module contains focused unit tests specifically for the /predict endpoint
as required by the ML Engineer Debugging Challenge.

Tests cover:
- Basic functionality 
- Input validation
- Response format validation
- Error handling
- Sentiment accuracy validation
"""

import pytest
import requests
import json
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://127.0.0.1:8080"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"


class TestPredictEndpointBasic:
    """Basic unit tests for the /predict endpoint."""
    
    def test_predict_positive_sentiment(self):
        """Test /predict with clearly positive text."""
        test_data = {"text": "This movie is absolutely amazing and wonderful!"}
        
        response = requests.post(PREDICT_ENDPOINT, json=test_data, timeout=10)
        
        # Skip if model not loaded
        if response.status_code == 503:
            pytest.skip("Model not loaded. Run: python -m ml.train")
            
        # Basic response validation
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Parse response
        data = response.json()
        
        # Response structure validation
        assert isinstance(data, dict)
        assert "label" in data
        assert "confidence" in data
        assert len(data) == 2  # Only label and confidence expected
        
        # Value validation
        assert data["label"] == "positive"
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["confidence"] > 0.5  # Should be confident about positive sentiment
        
        print(f"✓ Positive test passed: {data}")

    def test_predict_negative_sentiment(self):
        """Test /predict with clearly negative text."""
        test_data = {"text": "This movie is terrible, awful, and completely boring."}
        
        response = requests.post(PREDICT_ENDPOINT, json=test_data, timeout=10)
        
        # Skip if model not loaded
        if response.status_code == 503:
            pytest.skip("Model not loaded. Run: python -m ml.train")
            
        # Basic response validation
        assert response.status_code == 200
        
        # Parse response
        data = response.json()
        
        # Response structure validation
        assert isinstance(data, dict)
        assert "label" in data
        assert "confidence" in data
        
        # Value validation
        assert data["label"] == "negative"
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["confidence"] > 0.5  # Should be confident about negative sentiment
        
        print(f"✓ Negative test passed: {data}")

    def test_predict_response_format(self):
        """Test that /predict returns correct response format."""
        test_data = {"text": "Great movie!"}
        
        response = requests.post(PREDICT_ENDPOINT, json=test_data, timeout=10)
        
        # Skip if model not loaded
        if response.status_code == 503:
            pytest.skip("Model not loaded. Run: python -m ml.train")
            
        assert response.status_code == 200
        
        # Check response is valid JSON
        try:
            data = response.json()
        except json.JSONDecodeError:
            pytest.fail("Response is not valid JSON")
        
        # Check required fields exist
        required_fields = ["label", "confidence"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["label"], str), "Label must be string"
        assert isinstance(data["confidence"], (int, float)), "Confidence must be numeric"
        
        # Check valid values
        assert data["label"] in ["positive", "negative"], f"Invalid label: {data['label']}"
        assert 0.0 <= data["confidence"] <= 1.0, f"Invalid confidence: {data['confidence']}"
        
        print(f"✓ Response format test passed: {data}")

    def test_predict_input_validation(self):
        """Test /predict input validation."""
        
        # Test empty text
        response = requests.post(PREDICT_ENDPOINT, json={"text": ""}, timeout=5)
        assert response.status_code == 422  # Validation error
        
        # Test missing text field
        response = requests.post(PREDICT_ENDPOINT, json={}, timeout=5)
        assert response.status_code == 422  # Validation error
        
        # Test invalid JSON
        response = requests.post(
            PREDICT_ENDPOINT, 
            data="invalid json", 
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        assert response.status_code == 422  # Validation error
        
        # Test non-string text
        response = requests.post(PREDICT_ENDPOINT, json={"text": 123}, timeout=5)
        assert response.status_code == 422  # Validation error
        
        print("✓ Input validation tests passed")

    def test_predict_various_sentiments(self):
        """Test /predict with various sentiment examples."""
        test_cases = [
            # (text, expected_label)
            ("I love this!", "positive"),
            ("This is fantastic!", "positive"),
            ("Amazing quality!", "positive"),
            ("I hate this.", "negative"),
            ("This is awful.", "negative"),
            ("Completely terrible.", "negative"),
        ]
        
        results = []
        
        for text, expected_label in test_cases:
            response = requests.post(PREDICT_ENDPOINT, json={"text": text}, timeout=10)
            
            # Skip if model not loaded
            if response.status_code == 503:
                pytest.skip("Model not loaded. Run: python -m ml.train")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate structure
            assert "label" in data
            assert "confidence" in data
            assert data["label"] in ["positive", "negative"]
            assert 0.0 <= data["confidence"] <= 1.0
            
            # Check if prediction matches expectation (should be correct for obvious cases)
            if data["confidence"] > 0.7:  # Only check high-confidence predictions
                assert data["label"] == expected_label, \
                    f"Expected '{expected_label}' for '{text}', got '{data['label']}'"
            
            results.append({
                "text": text,
                "expected": expected_label,
                "predicted": data["label"],
                "confidence": data["confidence"]
            })
        
        print("✓ Various sentiments test results:")
        for result in results:
            status = "✓" if result["expected"] == result["predicted"] else "✗"
            print(f"  {status} '{result['text']}' → {result['predicted']} ({result['confidence']:.3f})")

    def test_predict_edge_cases(self):
        """Test /predict with edge cases."""
        edge_cases = [
            "okay",  # Neutral
            ".",     # Single punctuation
            "a",     # Single character
            "The movie was not bad.",  # Double negative
            "This is a movie.",  # Neutral factual statement
        ]
        
        for text in edge_cases:
            response = requests.post(PREDICT_ENDPOINT, json={"text": text}, timeout=10)
            
            # Skip if model not loaded
            if response.status_code == 503:
                pytest.skip("Model not loaded. Run: python -m ml.train")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should still return valid format even for edge cases
            assert "label" in data
            assert "confidence" in data
            assert data["label"] in ["positive", "negative"]
            assert 0.0 <= data["confidence"] <= 1.0
        
        print("✓ Edge cases test passed")

    def test_predict_performance(self):
        """Test /predict response time."""
        import time
        
        test_data = {"text": "This is a performance test."}
        
        start_time = time.time()
        response = requests.post(PREDICT_ENDPOINT, json=test_data, timeout=10)
        end_time = time.time()
        
        # Skip if model not loaded
        if response.status_code == 503:
            pytest.skip("Model not loaded. Run: python -m ml.train")
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds
        
        print(f"✓ Performance test passed: {response_time:.3f}s response time")


def test_predict_endpoint_availability():
    """Test that the /predict endpoint is available."""
    try:
        # Just check if endpoint exists (any response means it's available)
        response = requests.post(PREDICT_ENDPOINT, json={"text": "test"}, timeout=5)
        # Any response code (200, 422, 503) means endpoint exists
        assert response.status_code in [200, 422, 503]
        print("✓ Endpoint availability test passed")
    except requests.exceptions.ConnectionError:
        pytest.fail("Server is not running. Start with: uvicorn app.main:app --port 8080")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BASIC UNIT TESTS for /predict ENDPOINT")
    print("="*60)
    print("\nThese tests validate:")
    print("• Basic functionality of /predict endpoint")
    print("• Input validation and error handling") 
    print("• Response format correctness")
    print("• Sentiment prediction accuracy")
    print("• Performance characteristics")
    print("\nTo run these tests:")
    print("1. Start server: uvicorn app.main:app --port 8080")
    print("2. Run tests: pytest tests/test_predict_basic.py -v")
    print("="*60)
    
    # Run the tests
    pytest.main([__file__, "-v"])
