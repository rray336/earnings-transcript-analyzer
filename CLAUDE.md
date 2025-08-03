# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Earnings Transcript Analyzer** - a Python application that analyzes earnings call transcripts (PDF format) to extract sentiment, categorize questions, assess call strength, and identify analyst concerns. The tool can also build predictive models for stock price movements when combined with earnings performance data.

## Development Setup

### Dependencies Installation
```bash
pip install -r requirements.txt
```

Required packages include:
- pandas, numpy for data processing
- pdfplumber for PDF text extraction  
- textblob, vaderSentiment for sentiment analysis
- matplotlib, seaborn for visualizations
- scikit-learn for predictive modeling
- openpyxl for Excel file handling
- flask, werkzeug for web interface

### Running the Application

**Web Interface (Recommended):**
```bash
python app.py
```
Open browser to `http://localhost:5000` for drag-and-drop PDF upload interface.

**Command Line:**
```bash
python earnings_analyzer.py
```
The main script is configured to analyze transcripts in a specific folder structure. You'll need to update the paths in the `main()` function (lines 657-658) to point to your transcript folder and optional Excel file.

## Code Architecture

### Core Class: EarningsTranscriptAnalyzer

**Initialization & Data Loading:**
- `__init__()` - Sets up analyzer with PDF folder path and optional Excel earnings data
- `load_earnings_data()` - Loads Excel file with earnings performance metrics
- `extract_text_from_pdf()` - Extracts text content from PDF transcripts

**Text Processing & Analysis:**
- `parse_transcript_sections()` - Separates prepared remarks from Q&A sections
- `extract_questions()` - Uses regex to identify individual analyst questions
- `categorize_question()` - Classifies questions into predefined categories (guidance, margins, revenue, etc.)

**Sentiment & Strength Analysis:**
- `analyze_sentiment()` - Runs both TextBlob and VADER sentiment analysis
- `analyze_call_strength()` - Calculates confidence, uncertainty, and financial strength scores
- `calculate_segment_strength()` - Computes composite strength metrics for text segments
- `identify_analyst_concerns()` - Detects evasive language and defensive responses

**Machine Learning:**
- `build_predictive_model()` - Creates RandomForest model to predict stock movements
- `predict_stock_movement()` - Makes predictions on new transcript analysis
- Uses features like sentiment scores, call strength, concern ratios, and earnings performance

**Output & Visualization:**
- `create_visualizations()` - Generates dashboard with sentiment comparisons, question distributions
- `export_results()` - Outputs summary CSV/Excel files and detailed question analysis
- `create_summary_dataframe()` - Aggregates results across all analyzed transcripts

### Key Data Structures

**Question Categories:** Pre-defined keyword mapping for guidance, margins, revenue, competition, strategy, product, market, financial topics.

**Concern Indicators:** Detection patterns for hedging language, uncertainty, defensive responses, negative sentiment, and evasive answers.

**Analysis Results:** Each transcript analysis includes sentiment scores, question categorization, call strength metrics, and concern indicators.

## Data Requirements

The analyzer expects:
1. **PDF transcripts** in a designated folder
2. **Optional Excel file** with columns: Earnings, Earnings_Date, EPS_vs_Expectations, Guidance_vs_Expectations, Stock_Reaction

All analysis results are displayed in the web browser interface only - no files are generated.

## Model Features

When building predictive models, the system uses:
- Sentiment scores (overall, prepared remarks, Q&A)
- Call strength metrics (confidence, uncertainty, financial strength)
- Analyst concern ratios and indicators  
- Basic metrics (word count, question count)
- Earnings performance vs expectations
- Stock reaction as target variable