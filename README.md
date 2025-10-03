# Airline Incidents Analysis

## Project Overview
Analysis of airline incident reports to evaluate the effectiveness of current Part Failure classification systems and identify areas for improvement in maintenance documentation practices.

Kaggle dataset page:
https://www.kaggle.com/datasets/tarique7/airline-incidents-safety-data?select=Airline+Occurences.csv

## Business Objective
Rather than creating a new Classification System, this analysis audits existing "Part Failure" classifications to identify:
- Which failure categories are well-documented vs poorly documented
- Where inconsistent terminology creates classification confusion
- Priority areas for maintenance team training and standardisation (80/20 analysis)

**Target Audience**: Maintenance & Support (M&S) offices seeking data-driven insights for resource allocation and training priorities.

## Methodology
- **Text Mining**: TF-IDF analysis of incident report narratives
- **Classification Validation**: Machine Learning to assess prediction confidence by Failure type
- **Gap Analysis**: Identification of inconsistent documentation patterns

## Key Limitation
This analysis does not include aviation domain expertise validation. Results identify statistical patterns in documentation rather than technical failure analysis.

## Current Status
- [x] Data acquisition and preprocessing
- [x] TF-IDF model development and validation (82% accuracy)
- [x] Model comparison (Individual tokens outperform phrases)
- [x] Classification quality audit implementation
- [ ] SQL database schema design
- [ ] Power BI dasboard developoment

## Technical Stack
- **Development**: Python (PyCharm)
- **Analysis Documentation**: Jupyter Notebooks
- **Database**: Microsoft SQL Server
- **Business Intelligence**: Power BI
- **ML Libraries**: scikit-learn, pandas, sqlalchemy

## Final Deliverables
1. Classification System Health Report
2. SQL database with structured analysis results
3. Power BI exceutive dashboard
4. Actionable recommendations for M&S teams