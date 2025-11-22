#!/usr/bin/env python3
"""
Test script to verify the vLLM integration with the RAG system.
"""

import json

import requests

from backend.logging_config import get_logger

logger = get_logger(__name__)

def test_vllm_integration():
    """Test the integrated RAG + vLLM system."""
    base_url = "http://localhost:8001"  # assuming your RAG service runs on 8001
    
    # Test queries that should work with the sample data
    test_queries = [
        {
            "query": "What is our revenue growth?",
            "user_groups": ["finance", "executives"]
        },
        {
            "query": "Tell me about the AI chatbot performance",
            "user_groups": ["engineering", "product"]
        },
        {
            "query": "What security vulnerabilities were found?",
            "user_groups": ["security", "engineering"]
        }
    ]
    
    logger.info("Testing vLLM integration with RAG system...")
    
    for i, test_case in enumerate(test_queries, 1):
        logger.info(f"\n--- Test Case {i} ---")
        logger.info(f"Query: {test_case['query']}")
        logger.info(f"User Groups: {test_case['user_groups']}")
        
        try:
            response = requests.post(
                f"{base_url}/query",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=30  # vLLM might take some time
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Success! Response: {result['response'][:200]}...")
                logger.info(f"Sources found: {len(result['sources'])}")
                
                for j, source in enumerate(result['sources']):
                    logger.info(f"  Source {j+1}: {source['source']}")
            else:
                logger.error(f"Request failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            logger.error("Request timed out - vLLM might be slow or not responding")
        except Exception as e:
            logger.error(f"Error testing query: {e}")

def test_health_check():
    """Test the health endpoint."""
    try:
        response = requests.get("http://localhost:8001/health")
        if response.status_code == 200:
            logger.info(f"Health check passed: {response.json()}")
            return True
        else:
            logger.error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing vLLM + RAG integration...")
    
    # Test health endpoint first
    if test_health_check():
        # Test vLLM integration
        test_vllm_integration()
    else:
        logger.error("Health check failed - make sure the RAG service is running")
    
    logger.info("\nTesting completed!")
