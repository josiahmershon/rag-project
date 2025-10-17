#!/usr/bin/env python3
"""
Create sample documents with granular access control for testing.
Each document can have custom metadata, access groups, and security levels.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

def create_sample_documents_with_metadata() -> Path:
    """
    Create a structured set of sample documents with granular access control.
    Returns the directory path containing the documents.
    """
    
    # Create temporary directory structure
    temp_dir = Path(tempfile.mkdtemp(prefix="rag_sample_docs_"))
    logger.info(f"Creating sample documents in: {temp_dir}")
    
    # Define documents with granular access control
    documents = [
        {
            "path": "finance/quarterly_report_q4_2024.txt",
            "content": """
            FINANCIAL QUARTERLY REPORT - Q4 2024
            CONFIDENTIAL - FINANCE TEAM ONLY
            
            Executive Summary:
            Our Q4 2024 financial performance exceeded expectations with a 23% increase in revenue 
            compared to Q3. Total revenue reached $4.2M, driven primarily by enterprise client 
            acquisitions and successful product launches.
            
            Key Financial Metrics:
            - Revenue: $4.2M (+23% QoQ)
            - Gross Margin: 78% (+2% QoQ)
            - Operating Expenses: $2.1M
            - Net Profit: $1.8M (+31% QoQ)
            - Cash Position: $12.5M
            
            Department Performance:
            - Engineering: 45% of revenue
            - Sales: 35% of revenue
            - Marketing: 20% of revenue
            
            Outlook for Q1 2025:
            We project continued growth with revenue targets of $5.1M for Q1 2025.
            """,
            "metadata": {
                "department": "Finance",
                "security_level": "Confidential",
                "allowed_groups": ["finance", "executives", "cfo"],
                "document_type": "financial_report",
                "quarter": "Q4_2024",
                "tags": ["revenue", "profit", "quarterly", "confidential"]
            }
        },
        {
            "path": "engineering/technical_architecture_v2.md",
            "content": """
            # Technical Architecture v2.0
            
            ## System Overview
            Our new microservices architecture has been successfully deployed across all production environments.
            
            ## Key Components
            
            ### API Gateway
            - Handles 10,000+ requests per minute
            - 99.9% uptime achieved
            - Response time: <200ms average
            
            ### Database Layer
            - PostgreSQL with read replicas
            - Redis for caching
            - Automated backups every 6 hours
            
            ### Monitoring Stack
            - Prometheus for metrics
            - Grafana dashboards
            - AlertManager for notifications
            
            ## Performance Improvements
            - 40% reduction in API response times
            - 60% improvement in database query performance
            - 99.9% system uptime (up from 99.5%)
            
            ## Security Enhancements
            - OAuth 2.0 implementation
            - Rate limiting on all endpoints
            - Automated security scanning
            """,
            "metadata": {
                "department": "Engineering",
                "security_level": "Internal",
                "allowed_groups": ["engineering", "devops", "product", "tech_leads"],
                "document_type": "technical_documentation",
                "version": "2.0",
                "tags": ["architecture", "microservices", "performance", "security"]
            }
        },
        {
            "path": "hr/employee_handbook_2024.txt",
            "content": """
            EMPLOYEE HANDBOOK 2024
            INTERNAL USE ONLY
            
            Welcome to our company! This handbook outlines our policies, procedures, and expectations.
            
            WORKING HOURS & SCHEDULE
            - Standard hours: 9:00 AM - 5:00 PM, Monday-Friday
            - Flexible start times: 8:00 AM - 10:00 AM
            - Remote work: Up to 3 days per week with manager approval
            - Core collaboration hours: 10:00 AM - 3:00 PM
            
            VACATION & TIME OFF
            - Annual vacation: 15 days (increases with tenure)
            - Sick leave: 10 days per year
            - Personal days: 3 days per year
            - Holiday schedule: 12 company holidays
            
            PERFORMANCE & DEVELOPMENT
            - Annual reviews: December
            - Mid-year check-ins: June
            - Professional development budget: $2,000 per employee
            - Conference attendance: Up to 2 per year
            
            BENEFITS PACKAGE
            - Health insurance: Company pays 80%
            - Dental & vision: Company pays 100%
            - 401(k) matching: Up to 6% of salary
            - Life insurance: 2x annual salary
            - Disability insurance: 60% of salary
            
            CODE OF CONDUCT
            - Respectful communication
            - Zero tolerance for harassment
            - Confidentiality requirements
            - Conflict of interest policies
            """,
            "metadata": {
                "department": "Human Resources",
                "security_level": "Internal",
                "allowed_groups": ["hr", "employees", "management", "executives"],
                "document_type": "policy_document",
                "year": "2024",
                "tags": ["policies", "benefits", "handbook", "internal"]
            }
        },
        {
            "path": "security/incident_response_plan.pdf",
            "content": """
            SECURITY INCIDENT RESPONSE PLAN
            CONFIDENTIAL - SECURITY TEAM ONLY
            
            This document outlines our comprehensive incident response procedures for security breaches.
            
            INCIDENT CLASSIFICATION
            Level 1 - Low Impact: Minor security events, no data exposure
            Level 2 - Medium Impact: Potential data exposure, limited scope
            Level 3 - High Impact: Confirmed data breach, significant exposure
            Level 4 - Critical Impact: Major breach, regulatory implications
            
            RESPONSE TEAM ROLES
            - Incident Commander: CISO
            - Technical Lead: Senior Security Engineer
            - Communications Lead: PR Director
            - Legal Counsel: General Counsel
            - Executive Sponsor: CEO
            
            IMMEDIATE RESPONSE PROCEDURES
            1. Contain the incident (within 1 hour)
            2. Assess the scope and impact (within 4 hours)
            3. Notify stakeholders (within 8 hours)
            4. Begin forensic investigation (within 24 hours)
            5. Implement remediation (ongoing)
            
            COMMUNICATION PROTOCOLS
            - Internal: Slack #security-incidents channel
            - External: Press release template ready
            - Regulatory: 72-hour notification requirement
            - Customer: Email notification system
            
            RECOVERY PROCEDURES
            - System restoration protocols
            - Data recovery procedures
            - Security hardening measures
            - Post-incident review process
            
            CONTACT INFORMATION
            - Security Hotline: +1-555-SECURITY
            - Emergency Email: security@company.com
            - On-call Engineer: Rotating schedule
            """,
            "metadata": {
                "department": "Security",
                "security_level": "Confidential",
                "allowed_groups": ["security", "ciso", "executives", "legal"],
                "document_type": "security_procedure",
                "classification": "confidential",
                "tags": ["incident_response", "security", "procedures", "confidential"]
            }
        },
        {
            "path": "product/roadmap_2025.txt",
            "content": """
            PRODUCT ROADMAP 2025
            INTERNAL - PRODUCT TEAM
            
            Our 2025 product roadmap focuses on AI integration, user experience improvements, 
            and enterprise features.
            
            Q1 2025 - AI FOUNDATION
            - Implement AI-powered search
            - Deploy machine learning models
            - Launch beta testing program
            - Target: 1,000 beta users
            
            Q2 2025 - USER EXPERIENCE
            - Redesign user interface
            - Mobile app development
            - Performance optimization
            - Target: 50% faster load times
            
            Q3 2025 - ENTERPRISE FEATURES
            - Advanced analytics dashboard
            - Custom integrations API
            - White-label solutions
            - Target: 10 enterprise clients
            
            Q4 2025 - SCALING & GROWTH
            - International expansion
            - Multi-language support
            - Advanced security features
            - Target: 100,000 active users
            
            RESOURCE ALLOCATION
            - Engineering: 60% of team
            - Design: 20% of team
            - Product Management: 20% of team
            
            SUCCESS METRICS
            - User engagement: +40%
            - Customer satisfaction: 4.5/5
            - Revenue growth: +50%
            - Market share: 15%
            """,
            "metadata": {
                "department": "Product",
                "security_level": "Internal",
                "allowed_groups": ["product", "engineering", "design", "executives", "sales"],
                "document_type": "roadmap",
                "year": "2025",
                "tags": ["roadmap", "ai", "product", "strategy"]
            }
        },
        {
            "path": "marketing/campaign_results_q4.txt",
            "content": """
            MARKETING CAMPAIGN RESULTS - Q4 2024
            INTERNAL - MARKETING TEAM
            
            CAMPAIGN OVERVIEW
            Our Q4 marketing campaigns focused on enterprise lead generation and brand awareness.
            
            CAMPAIGN PERFORMANCE
            - Digital Ads: $50K spend, 2,500 leads, $20 cost per lead
            - Content Marketing: 15 blog posts, 50K page views, 2% conversion
            - Email Campaigns: 100K sends, 25% open rate, 5% click rate
            - Social Media: 500K impressions, 10K engagements, 2% engagement rate
            
            LEAD GENERATION RESULTS
            - Total leads: 3,200 (+15% vs Q3)
            - Qualified leads: 800 (+25% vs Q3)
            - Sales accepted leads: 400 (+30% vs Q3)
            - Conversion rate: 12.5% (industry average: 8%)
            
            BRAND AWARENESS METRICS
            - Brand mentions: +40% vs Q3
            - Website traffic: +25% vs Q3
            - Social media followers: +15% vs Q3
            - Search ranking: Top 3 for target keywords
            
            ROI ANALYSIS
            - Total marketing spend: $75K
            - Generated revenue: $2.1M
            - ROI: 2,800%
            - Customer acquisition cost: $187
            - Customer lifetime value: $15,000
            
            LESSONS LEARNED
            - Video content performs 3x better than text
            - Personalization increases conversion by 40%
            - Timing matters: Tuesday-Thursday best for B2B
            - LinkedIn most effective for enterprise leads
            """,
            "metadata": {
                "department": "Marketing",
                "security_level": "Internal",
                "allowed_groups": ["marketing", "sales", "executives", "product"],
                "document_type": "campaign_report",
                "quarter": "Q4_2024",
                "tags": ["marketing", "campaigns", "roi", "leads"]
            }
        }
    ]
    
    # Create directory structure and files
    for doc in documents:
        # Create directory if it doesn't exist
        file_path = temp_dir / doc["path"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write document content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(doc["content"])
        
        # Write metadata file
        metadata_path = file_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(doc["metadata"], f, indent=2)
        
        logger.info(f"Created: {file_path}")
        logger.info(f"  Metadata: {doc['metadata']}")
    
    # Create a metadata index file
    index_data = {
        "total_documents": len(documents),
        "departments": list(set(doc["metadata"]["department"] for doc in documents)),
        "security_levels": list(set(doc["metadata"]["security_level"] for doc in documents)),
        "all_groups": list(set(group for doc in documents for group in doc["metadata"]["allowed_groups"])),
        "documents": [
            {
                "path": doc["path"],
                "department": doc["metadata"]["department"],
                "security_level": doc["metadata"]["security_level"],
                "allowed_groups": doc["metadata"]["allowed_groups"],
                "document_type": doc["metadata"]["document_type"]
            }
            for doc in documents
        ]
    }
    
    index_path = temp_dir / "metadata_index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    
    logger.info(f"Created metadata index: {index_path}")
    logger.info(f"Total documents created: {len(documents)}")
    
    return temp_dir

def create_custom_document(
    content: str,
    department: str,
    security_level: str,
    allowed_groups: List[str],
    document_type: str = "custom",
    tags: List[str] = None,
    **kwargs
) -> Dict:
    """
    Create a custom document with specific metadata.
    
    Args:
        content: Document content
        department: Department (Finance, Engineering, HR, etc.)
        security_level: Security level (Internal, Confidential, Public)
        allowed_groups: List of allowed user groups
        document_type: Type of document
        tags: List of tags
        **kwargs: Additional metadata fields
    
    Returns:
        Document dictionary with content and metadata
    """
    metadata = {
        "department": department,
        "security_level": security_level,
        "allowed_groups": allowed_groups,
        "document_type": document_type,
        "tags": tags or [],
        **kwargs
    }
    
    return {
        "content": content,
        "metadata": metadata
    }

if __name__ == "__main__":
    # Create sample documents
    sample_dir = create_sample_documents_with_metadata()
    print(f"\nSample documents created in: {sample_dir}")
    print("\nDirectory structure:")
    
    for root, dirs, files in os.walk(sample_dir):
        level = root.replace(str(sample_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    print(f"\nTo test the ingestion pipeline:")
    print(f"python backend/ingestion_pipeline.py {sample_dir} --dry-run")
