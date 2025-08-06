#!/usr/bin/env python3
"""
Enhanced Model Training Script for TSN Earnings Analysis
This script trains an improved model using all available TSN transcripts and enhanced features.
"""

import os
import sys
from pathlib import Path
from earnings_analyzer import EarningsTranscriptAnalyzer
import pandas as pd

def main():
    print("Enhanced TSN Earnings Model Training")
    print("=" * 50)
    
    # Paths
    transcript_folder = r"C:\Users\rahul\OneDrive\IMP_DOCS\PORTFOLIO\STOCKS\CONSUMER HOUSING\TSN\Transcripts"
    excel_file = "TSN_Enhanced_Earnings_Data.xlsx"
    
    # Verify files exist
    if not os.path.exists(transcript_folder):
        print(f"[ERROR] Transcript folder not found: {transcript_folder}")
        return
    
    if not os.path.exists(excel_file):
        print(f"[ERROR] Excel file not found: {excel_file}")
        return
    
    # Get PDF files (exclude subfolders)
    pdf_files = []
    for file in os.listdir(transcript_folder):
        if file.endswith('.pdf') and os.path.isfile(os.path.join(transcript_folder, file)):
            pdf_files.append(file)
    
    print(f"Found {len(pdf_files)} PDF transcripts:")
    for pdf in sorted(pdf_files):
        print(f"   - {pdf}")
    
    # Load and display earnings data
    df = pd.read_excel(excel_file)
    print(f"\nEarnings data loaded:")
    print(f"   - {len(df)} quarters of data")
    print(f"   - Stock movements: {df['Stock_Reaction'].min():.1f}% to {df['Stock_Reaction'].max():.1f}%")
    print(f"   - Mean movement: {df['Stock_Reaction'].mean():.1f}%")
    
    # Initialize analyzer
    print(f"\nInitializing Enhanced Analyzer...")
    analyzer = EarningsTranscriptAnalyzer(
        pdf_folder_path=transcript_folder,
        excel_file_path=excel_file
    )
    
    # Analyze all transcripts
    print(f"\nAnalyzing {len(pdf_files)} transcripts...")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(transcript_folder, pdf_file)
        print(f"   Processing: {pdf_file}")
        result = analyzer.analyze_single_transcript(pdf_path)
        if result:
            analyzer.analysis_results[pdf_file] = result
    
    # Build enhanced predictive model
    print(f"\nBuilding Enhanced Predictive Model...")
    model_results = analyzer.build_predictive_model()
    
    if model_results:
        print(f"\n[SUCCESS] Model Training Complete!")
        print(f"   - R² Score: {model_results['r2']:.3f}")
        print(f"   - MSE: {model_results['mse']:.3f}")
        if hasattr(model_results['model'], 'oob_score_'):
            print(f"   - OOB Score: {model_results['model'].oob_score_:.3f}")
        
        print(f"\nTop 10 Feature Importance:")
        feature_imp = model_results['feature_importance'].head(10)
        for idx, row in feature_imp.iterrows():
            print(f"   {idx+1:2d}. {row['feature']:<25} {row['importance']*100:5.1f}%")
        
        # Save model
        from app import save_trained_model
        if save_trained_model(model_results):
            print(f"\nModel saved successfully!")
        else:
            print(f"\n[ERROR] Failed to save model")
        
        print(f"\nEnhanced model ready for predictions!")
        
    else:
        print(f"\n[ERROR] Model training failed")

if __name__ == "__main__":
    main()