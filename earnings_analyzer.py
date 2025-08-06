import os
import pandas as pd
import pdfplumber
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# For sentiment analysis
from textblob import TextBlob
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER not installed. Install with: pip install vaderSentiment")

class EarningsTranscriptAnalyzer:
    def __init__(self, pdf_folder_path, excel_file_path=None):
        """
        Initialize the analyzer with a folder containing PDF transcripts and optional Excel data
        
        Args:
            pdf_folder_path (str): Path to folder containing PDF files
            excel_file_path (str): Path to Excel file with earnings data
        """
        self.pdf_folder = Path(pdf_folder_path)
        self.excel_file_path = excel_file_path
        self.transcripts = {}
        self.analysis_results = {}
        self.earnings_data = None
        
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Question categories for classification
        self.question_categories = {
            'guidance': ['guidance', 'outlook', 'forecast', 'expect', 'target', 'goal'],
            'margins': ['margin', 'profitability', 'cost', 'expense', 'efficiency'],
            'revenue': ['revenue', 'sales', 'growth', 'top line', 'demand'],
            'competition': ['competitor', 'competitive', 'market share', 'rival'],
            'strategy': ['strategy', 'strategic', 'plan', 'initiative', 'focus'],
            'product': ['product', 'launch', 'development', 'innovation', 'feature'],
            'market': ['market', 'industry', 'sector', 'trend', 'environment'],
            'financial': ['balance sheet', 'cash flow', 'debt', 'capital', 'investment']
        }
        
        # Concern indicators for gap analysis
        self.concern_indicators = {
            'hedging': ['however', 'but', 'although', 'despite', 'while', 'cautious'],
            'uncertainty': ['uncertain', 'unclear', 'challenging', 'difficult', 'volatile'],
            'defensive': ['manage', 'control', 'monitor', 'watch', 'careful'],
            'negative': ['decline', 'decrease', 'drop', 'fall', 'lower', 'weak'],
            'evasive': ['I think', 'maybe', 'probably', 'we believe', 'as I said']
        }
        
        # Load earnings data if provided
        if self.excel_file_path and os.path.exists(self.excel_file_path):
            self.load_earnings_data()
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return None
    
    def parse_transcript_sections(self, text):
        """
        Parse transcript into different sections (prepared remarks vs Q&A)
        """
        # Common patterns to identify Q&A section
        qa_indicators = [
            r"question.*answer",
            r"q&a",
            r"questions.*answers",
            r"operator.*questions"
        ]
        
        # Find Q&A section start
        qa_start = None
        text_lower = text.lower()
        
        for pattern in qa_indicators:
            match = re.search(pattern, text_lower)
            if match:
                qa_start = match.start()
                break
        
        if qa_start:
            prepared_remarks = text[:qa_start]
            qa_section = text[qa_start:]
        else:
            # If no clear Q&A section found, assume second half is Q&A
            midpoint = len(text) // 2
            prepared_remarks = text[:midpoint]
            qa_section = text[midpoint:]
        
        return {
            'prepared_remarks': prepared_remarks,
            'qa_section': qa_section,
            'full_text': text
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using both TextBlob and VADER"""
        results = {}
        
        # TextBlob analysis
        blob = TextBlob(text)
        results['textblob_polarity'] = blob.sentiment.polarity
        results['textblob_subjectivity'] = blob.sentiment.subjectivity
        
        # VADER analysis (if available)
        if VADER_AVAILABLE:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            results.update({f'vader_{k}': v for k, v in vader_scores.items()})
        
        return results
    
    def load_earnings_data(self):
        """Load earnings data from Excel file"""
        try:
            self.earnings_data = pd.read_excel(self.excel_file_path)
            # Expected columns: Earnings, Earnings_Date, EPS_vs_Expectations, Guidance_vs_Expectations, Stock_Reaction
            expected_cols = ['Earnings', 'Earnings_Date', 'EPS_vs_Expectations', 'Guidance_vs_Expectations', 'Stock_Reaction']
            
            if len(self.earnings_data.columns) >= 5:
                self.earnings_data.columns = expected_cols[:len(self.earnings_data.columns)]
                print(f"Loaded earnings data with {len(self.earnings_data)} records")
            else:
                print("Warning: Excel file has fewer than 5 columns. Please check format.")
                
        except Exception as e:
            print(f"Error loading earnings data: {str(e)}")
            self.earnings_data = None
    
    def analyze_call_strength(self, sections):
        """Analyze the strength of earnings call components"""
        strength_metrics = {}
        
        # Prepared remarks strength
        prepared_text = sections['prepared_remarks']
        prepared_strength = self.calculate_segment_strength(prepared_text, is_prepared=True)
        strength_metrics['prepared_remarks'] = prepared_strength
        
        # Q&A strength
        qa_text = sections['qa_section']
        qa_strength = self.calculate_segment_strength(qa_text, is_prepared=False)
        strength_metrics['qa_section'] = qa_strength
        
        # Overall call strength (weighted)
        overall_strength = (prepared_strength['composite_score'] * 0.4 + 
                          qa_strength['composite_score'] * 0.6)
        strength_metrics['overall_strength'] = overall_strength
        
        return strength_metrics
    
    def calculate_segment_strength(self, text, is_prepared=True):
        """Calculate strength metrics for a text segment"""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Confidence indicators
        confidence_words = ['confident', 'strong', 'solid', 'robust', 'optimistic', 
                          'pleased', 'excited', 'positive', 'growth', 'improve']
        confidence_score = sum(1 for word in confidence_words if word in text_lower) / len(words) * 100
        
        # Uncertainty indicators
        uncertainty_words = ['uncertain', 'challenging', 'difficult', 'cautious', 
                           'concerned', 'worried', 'volatile', 'pressure']
        uncertainty_score = sum(1 for word in uncertainty_words if word in text_lower) / len(words) * 100
        
        # Financial strength indicators
        financial_strength = ['revenue growth', 'margin expansion', 'cash generation', 
                            'market share', 'profitability', 'efficiency']
        financial_score = sum(1 for phrase in financial_strength if phrase in text_lower) / len(words) * 100
        
        # Calculate composite strength score
        if is_prepared:
            # For prepared remarks, emphasize confidence and financial metrics
            composite_score = (confidence_score * 0.4 + financial_score * 0.4 - uncertainty_score * 0.2)
        else:
            # For Q&A, emphasize how well they handle uncertainty
            composite_score = (confidence_score * 0.3 + financial_score * 0.3 - uncertainty_score * 0.4)
        
        return {
            'confidence_score': confidence_score,
            'uncertainty_score': uncertainty_score,
            'financial_score': financial_score,
            'composite_score': max(0, composite_score)  # Ensure non-negative
        }
    
    def identify_analyst_concerns(self, questions):
        """Identify potential gaps that could concern analysts"""
        concerns = {
            'evasive_responses': 0,
            'defensive_tone': 0,
            'hedging_language': 0,
            'uncertainty_indicators': 0,
            'concern_categories': Counter()
        }
        
        for question in questions:
            q_text = question['text'].lower()
            
            # Check for concern indicators
            for concern_type, indicators in self.concern_indicators.items():
                indicator_count = sum(1 for indicator in indicators if indicator in q_text)
                if indicator_count > 0:
                    concerns[f'{concern_type}_responses'] = concerns.get(f'{concern_type}_responses', 0) + 1
                    concerns['concern_categories'][concern_type] += indicator_count
        
        # Calculate overall concern score
        total_concerns = sum(v for k, v in concerns.items() if k.endswith('_responses'))
        concern_ratio = total_concerns / len(questions) if questions else 0
        concerns['overall_concern_ratio'] = concern_ratio
        
        return concerns
    
    def extract_questions(self, qa_text):
        """Extract individual questions from Q&A section"""
        # Pattern to identify questions (usually start with analyst name and company)
        question_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+.*?[:\-].*?)(?=[A-Z][a-z]+ [A-Z][a-z]+.*?[:\-]|$)'
        questions = re.findall(question_pattern, qa_text, re.DOTALL)
        
        # Clean up questions
        cleaned_questions = []
        for q in questions:
            # Remove extra whitespace and newlines
            cleaned = re.sub(r'\s+', ' ', q.strip())
            if len(cleaned) > 50:  # Filter out very short matches
                cleaned_questions.append(cleaned)
        
        return cleaned_questions
    
    def categorize_question(self, question_text):
        """Categorize questions based on keywords"""
        question_lower = question_text.lower()
        categories = []
        
        for category, keywords in self.question_categories.items():
            if any(keyword in question_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['general']
    
    def analyze_single_transcript(self, pdf_path):
        """Analyze a single earnings transcript"""
        if isinstance(pdf_path, str):
            filename = os.path.basename(pdf_path)
        else:
            filename = pdf_path.name
        print(f"Analyzing: {filename}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        
        # Parse sections
        sections = self.parse_transcript_sections(text)
        
        # Analyze sentiment for each section
        prepared_sentiment = self.analyze_sentiment(sections['prepared_remarks'])
        qa_sentiment = self.analyze_sentiment(sections['qa_section'])
        overall_sentiment = self.analyze_sentiment(sections['full_text'])
        
        # Extract and analyze questions
        questions = self.extract_questions(sections['qa_section'])
        question_analysis = []
        
        for question in questions:
            q_sentiment = self.analyze_sentiment(question)
            q_categories = self.categorize_question(question)
            
            question_analysis.append({
                'text': question[:200] + '...' if len(question) > 200 else question,
                'categories': q_categories,
                'sentiment': q_sentiment
            })
        
        # Analyze call strength
        call_strength = self.analyze_call_strength(sections)
        
        # Identify analyst concerns
        analyst_concerns = self.identify_analyst_concerns(question_analysis)
        
        # Compile results
        results = {
            'file_name': filename,
            'word_count': len(sections['full_text'].split()),
            'prepared_remarks_sentiment': prepared_sentiment,
            'qa_sentiment': qa_sentiment,
            'overall_sentiment': overall_sentiment,
            'question_count': len(questions),
            'questions': question_analysis,
            'question_categories': Counter([cat for q in question_analysis for cat in q['categories']]),
            'call_strength': call_strength,
            'analyst_concerns': analyst_concerns
        }
        
        return results
    
    def process_all_transcripts(self):
        """Process all PDF files in the specified folder"""
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_folder}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process...")
        
        for pdf_file in pdf_files:
            result = self.analyze_single_transcript(pdf_file)
            if result:
                self.analysis_results[pdf_file.stem] = result
        
        print(f"Successfully analyzed {len(self.analysis_results)} transcripts")
    
    def create_summary_dataframe(self):
        """Create a summary DataFrame of all analyses"""
        if not self.analysis_results:
            print("No analysis results available. Run process_all_transcripts() first.")
            return None
        
        summary_data = []
        
        for file_key, results in self.analysis_results.items():
            row = {
                'file_name': results['file_name'],
                'word_count': results['word_count'],
                'question_count': results['question_count'],
                'overall_textblob_polarity': results['overall_sentiment']['textblob_polarity'],
                'overall_textblob_subjectivity': results['overall_sentiment']['textblob_subjectivity'],
                'prepared_remarks_polarity': results['prepared_remarks_sentiment']['textblob_polarity'],
                'qa_polarity': results['qa_sentiment']['textblob_polarity'],
            }
            
            # Add VADER scores if available
            if VADER_AVAILABLE:
                row.update({
                    'overall_vader_compound': results['overall_sentiment']['vader_compound'],
                    'prepared_remarks_vader': results['prepared_remarks_sentiment']['vader_compound'],
                    'qa_vader': results['qa_sentiment']['vader_compound']
                })
            
            # Add question category counts
            for category in self.question_categories.keys():
                row[f'{category}_questions'] = results['question_categories'].get(category, 0)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def build_predictive_model(self):
        """Build a predictive model for stock price movement"""
        if not self.analysis_results or self.earnings_data is None:
            print("Need both transcript analysis and earnings data for modeling")
            return None
        
        # Prepare feature matrix
        model_data = []
        
        for file_key, results in self.analysis_results.items():
            # Find matching earnings data (you may need to adjust matching logic)
            earnings_row = None
            for idx, row in self.earnings_data.iterrows():
                # Simple matching - you may want to improve this
                if str(row.get('Earnings', '')).lower() in file_key.lower():
                    earnings_row = row
                    break
            
            if earnings_row is not None:
                # Extract enhanced features for better prediction
                features = {
                    # Core sentiment features
                    'overall_sentiment': results['overall_sentiment']['textblob_polarity'],
                    'prepared_sentiment': results['prepared_remarks_sentiment']['textblob_polarity'],
                    'qa_sentiment': results['qa_sentiment']['textblob_polarity'],
                    
                    # Call strength features
                    'overall_strength': results['call_strength']['overall_strength'],
                    'prepared_strength': results['call_strength']['prepared_remarks']['composite_score'],
                    'qa_strength': results['call_strength']['qa_section']['composite_score'],
                    
                    # Enhanced sentiment analysis
                    'sentiment_volatility': abs(results['prepared_remarks_sentiment']['textblob_polarity'] - results['qa_sentiment']['textblob_polarity']),
                    'sentiment_trend': results['qa_sentiment']['textblob_polarity'] - results['prepared_remarks_sentiment']['textblob_polarity'],
                    
                    # Question analysis enhancements
                    'questions_per_1000_words': (results['question_count'] / results['word_count']) * 1000 if results['word_count'] > 0 else 0,
                    'guidance_question_ratio': results['question_categories'].get('guidance', 0) / max(results['question_count'], 1),
                    'margin_question_ratio': results['question_categories'].get('margins', 0) / max(results['question_count'], 1),
                    
                    # Concern features
                    'concern_ratio': results['analyst_concerns']['overall_concern_ratio'],
                    'evasive_responses': results['analyst_concerns'].get('evasive_responses', 0),
                    'uncertainty_indicators': results['analyst_concerns'].get('uncertainty_responses', 0),
                    
                    # Basic metrics
                    'question_count': results['question_count'],
                    'word_count': results['word_count'],
                    
                    # Earnings performance
                    'eps_vs_expectations': self.encode_performance(earnings_row.get('EPS_vs_Expectations', 'Meet')),
                    'guidance_vs_expectations': self.encode_performance(earnings_row.get('Guidance_vs_Expectations', 'Meet')),
                    
                    # Target variable
                    'stock_reaction': self.parse_stock_reaction(earnings_row.get('Stock_Reaction', 0))
                }
                
                # Add VADER scores if available
                if VADER_AVAILABLE:
                    features['overall_vader'] = results['overall_sentiment']['vader_compound']
                    features['prepared_vader'] = results['prepared_remarks_sentiment']['vader_compound']
                    features['qa_vader'] = results['qa_sentiment']['vader_compound']
                
                model_data.append(features)
        
        if len(model_data) < 3:
            print("Need at least 3 data points for modeling")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(model_data)
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != 'stock_reaction']
        X = df[feature_cols]
        y = df['stock_reaction']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        if len(df) > 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Enhanced model training with better hyperparameters
        print(f"Training enhanced model with {len(feature_cols)} features and {len(df)} data points")
        
        # Use more sophisticated Random Forest configuration
        model = RandomForestRegressor(
            n_estimators=200,           # More trees for better accuracy
            max_depth=4,                # Prevent overfitting with small dataset
            min_samples_split=2,        # Allow more granular splits
            min_samples_leaf=1,         # Allow single-sample leaves
            max_features='sqrt',        # Feature subsampling
            random_state=42,
            bootstrap=True,             # Bootstrap sampling
            oob_score=True              # Out-of-bag score for validation
        )
        model.fit(X_train_scaled, y_train)
        
        print(f"Model trained. OOB Score: {model.oob_score_:.3f}")
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        model_results = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'mse': mse,
            'r2': r2,
            'feature_importance': feature_importance,
            'predictions': pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred
            })
        }
        
        print(f"Model Performance:")
        print(f"R² Score: {r2:.3f}")
        print(f"MSE: {mse:.3f}")
        print("\\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        return model_results
    
    def encode_performance(self, performance):
        """Encode performance strings to numeric values"""
        if pd.isna(performance):
            return 0
        performance_str = str(performance).lower()
        if 'beat' in performance_str:
            return 1
        elif 'miss' in performance_str:
            return -1
        else:  # meet
            return 0
    
    def parse_stock_reaction(self, reaction):
        """Parse stock reaction to numeric value"""
        if pd.isna(reaction):
            return 0
        
        # If it's already a number, return it
        try:
            return float(reaction)
        except:
            pass
        
        # If it's a string with percentage
        reaction_str = str(reaction).replace('%', '').replace('+', '')
        try:
            return float(reaction_str)
        except:
            return 0
    
    def predict_stock_movement(self, model_results, new_analysis):
        """Predict stock movement for new analysis"""
        if not model_results:
            return None
        
        # Extract same features as training
        features = []
        for col in model_results['feature_cols']:
            if col in new_analysis:
                features.append(new_analysis[col])
            else:
                features.append(0)  # Default value for missing features
        
        # Scale and predict
        features_scaled = model_results['scaler'].transform([features])
        prediction = model_results['model'].predict(features_scaled)[0]
        
        return prediction
    
    def create_visualizations(self, output_folder="analysis_output"):
        """Create visualizations of the analysis results"""
        print("Visualization functionality has been removed.")
        return None
    
    def export_results(self, output_folder="analysis_output"):
        """Export functionality disabled - results only displayed in browser"""
        print("File export disabled - results displayed in browser only")
        return

# Example usage
def main():
    # Set your paths here
    PDF_FOLDER = r"C:\Users\rahul\OneDrive\IMP_DOCS\PORTFOLIO\STOCKS\CONSUMER HOUSING\TSN\Transcripts"
    EXCEL_FILE = os.path.join(PDF_FOLDER, "EarningsGuidance.xlsx")

    # Initialize analyzer with Excel data
    analyzer = EarningsTranscriptAnalyzer(PDF_FOLDER, EXCEL_FILE)
    
    # Process all transcripts
    analyzer.process_all_transcripts()
    
    # Create summary DataFrame
    summary = analyzer.create_summary_dataframe()
    if summary is not None:
        print("\nSummary Statistics:")
        print(summary.describe())
    
    # Build predictive model if earnings data is available
    if analyzer.earnings_data is not None:
        print("\n" + "="*50)
        print("BUILDING PREDICTIVE MODEL")
        print("="*50)
        model_results = analyzer.build_predictive_model()
        
        if model_results:
            print(f"\nModel built successfully!")
            print("Use predict_stock_movement() to make predictions on new transcripts")
    
    # File output disabled for web interface
    
    print("\nAnalysis complete! Results available in web interface only.")
    
    # Display call strength summary
    if analyzer.analysis_results:
        print("\n" + "="*50) 
        print("EARNINGS CALL STRENGTH SUMMARY")
        print("="*50)
        for file_key, results in analyzer.analysis_results.items():
            strength = results['call_strength']
            concerns = results['analyst_concerns']
            print(f"\n{results['file_name']}:")
            print(f"  Overall Strength: {strength['overall_strength']:.2f}")
            print(f"  Prepared Remarks: {strength['prepared_remarks']['composite_score']:.2f}")
            print(f"  Q&A Strength: {strength['qa_section']['composite_score']:.2f}")
            print(f"  Analyst Concern Ratio: {concerns['overall_concern_ratio']:.2f}")
            
            # Highlight potential red flags
            if concerns['overall_concern_ratio'] > 0.3:
                print(f"  WARNING: HIGH CONCERN LEVEL - {concerns['overall_concern_ratio']:.1%} of responses show concern indicators")
            if strength['overall_strength'] < 2.0:
                print(f"  WARNING: LOW CALL STRENGTH - Overall strength below 2.0")

if __name__ == "__main__":
    main()