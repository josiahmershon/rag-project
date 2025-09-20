#!/usr/bin/env python3
"""
test script to verify the RAG retrieval functionality.
"""

import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_query_endpoint():
    """test the /query endpoint with sample queries."""
    base_url = "http://localhost:8001"
    
    test_queries = [
        {
            "query": "What is our revenue growth?",
            "user_groups": ["finance", "executives"]
        },
        {
            "query": "Tell me about the AI chatbot",
            "user_groups": ["engineering", "product"]
        },
        {
            "query": "What security issues do we have?",
            "user_groups": ["security", "engineering"]
        },
        {
            "query": "How are employees feeling?",
            "user_groups": ["hr", "management"]
        },
        {
            "query": "What machine learning models do we have?",
            "user_groups": ["data_science", "engineering"]
        },
        {
            "query": "Tell me about office construction",
            "user_groups": ["facilities", "management"]
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        logger.info(f"\n--- Test Case {i} ---")
        logger.info(f"Query: {test_case['query']}")
        logger.info(f"User Groups: {test_case['user_groups']}")
        
        try:
            response = requests.post(
                f"{base_url}/query",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Response: {result['response'][:300]}...")
                logger.info(f"Sources found: {len(result['sources'])}")
                
                for j, source in enumerate(result['sources']):
                    logger.info(f"  Source {j+1}: {source['source']}")
                    logger.info(f"    Text: {source['text'][:100]}...")
            else:
                logger.error(f"Request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error testing query: {e}")

def test_health_endpoint():
    """test the health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            logger.info(f"Health check passed: {response.json()}")
        else:
            logger.error(f"Health check failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Health check error: {e}")

if __name__ == "__main__":
    logger.info("Testing RAG retrieval system...")
    
    # test health endpoint first
    test_health_endpoint()
    
    # test query endpoint
    test_query_endpoint()
    
    logger.info("\nTesting completed!")
