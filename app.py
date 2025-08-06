from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import tempfile
from pathlib import Path
from werkzeug.utils import secure_filename
from earnings_analyzer import EarningsTranscriptAnalyzer
import json
from datetime import datetime
import pickle
import joblib

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'trained_models'
ALLOWED_EXTENSIONS = {'pdf'}
ALLOWED_EXCEL_EXTENSIONS = {'xlsx', 'xls'}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_excel_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXCEL_EXTENSIONS

def save_trained_model(model_results):
    """Save the trained model to disk"""
    try:
        model_path = os.path.join(MODEL_FOLDER, 'trained_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_results, f)
        
        # Save metadata
        metadata = {
            'trained_date': datetime.now().isoformat(),
            'r2_score': model_results['r2'],
            'mse': model_results['mse'],
            'feature_count': len(model_results['feature_cols'])
        }
        metadata_path = os.path.join(MODEL_FOLDER, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_trained_model():
    """Load the trained model from disk"""
    try:
        model_path = os.path.join(MODEL_FOLDER, 'trained_model.pkl')
        metadata_path = os.path.join(MODEL_FOLDER, 'model_metadata.json')
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            return None, None
        
        with open(model_path, 'rb') as f:
            model_results = pickle.load(f)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model_results, metadata
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def has_trained_model():
    """Check if a trained model exists"""
    model_path = os.path.join(MODEL_FOLDER, 'trained_model.pkl')
    metadata_path = os.path.join(MODEL_FOLDER, 'model_metadata.json')
    return os.path.exists(model_path) and os.path.exists(metadata_path)

@app.route('/')
def index():
    model_available = has_trained_model()
    return render_template('index.html', model_available=model_available)

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        flash('No files selected')
        return redirect(request.url)
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        flash('No files selected')
        return redirect(url_for('index'))
    
    uploaded_files = []
    temp_dir = tempfile.mkdtemp()
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            uploaded_files.append(file_path)
        else:
            flash(f'Invalid file type: {file.filename}. Only PDF files are allowed.')
            return redirect(url_for('index'))
    
    if not uploaded_files:
        flash('No valid PDF files uploaded')
        return redirect(url_for('index'))
    
    # Process the files
    try:
        analyzer = EarningsTranscriptAnalyzer(temp_dir)
        analyzer.process_all_transcripts()
        
        if not analyzer.analysis_results:
            flash('No analysis results generated. Please check your PDF files.')
            return redirect(url_for('index'))
        
        # Try to use existing trained model or build new one
        model_results = None
        model_available = False
        
        # First, try to load existing trained model
        saved_model, model_metadata = load_trained_model()
        if saved_model:
            try:
                # Use saved model to make predictions on new analysis
                model_results = use_saved_model_for_predictions(saved_model, analyzer.analysis_results)
                model_available = True
                print(f"Using saved model trained on {model_metadata['trained_date']}")
            except Exception as e:
                print(f"Error using saved model: {e}")
                
        # If no saved model or error, try to build new one if earnings data available
        if not model_available and analyzer.earnings_data is not None:
            try:
                model_results = analyzer.build_predictive_model()
                model_available = True
            except Exception as e:
                print(f"Model building failed: {e}")
        
        # Convert results for web display (no file output)
        results = prepare_results_for_web(analyzer.analysis_results, model_results, model_available)
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return render_template('results.html', results=results)
        
    except Exception as e:
        flash(f'Error processing files: {str(e)}')
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return redirect(url_for('index'))

def use_saved_model_for_predictions(saved_model, analysis_results):
    """Use a saved model to make predictions on new analysis results"""
    from earnings_analyzer import EarningsTranscriptAnalyzer
    
    # Create a dummy analyzer instance to use the prediction method
    dummy_analyzer = EarningsTranscriptAnalyzer("dummy_path")
    
    predictions = []
    for file_key, results in analysis_results.items():
        # Extract features for prediction
        features = {
            'overall_sentiment': results['overall_sentiment']['textblob_polarity'],
            'prepared_sentiment': results['prepared_remarks_sentiment']['textblob_polarity'],
            'qa_sentiment': results['qa_sentiment']['textblob_polarity'],
            'overall_strength': results['call_strength']['overall_strength'],
            'prepared_strength': results['call_strength']['prepared_remarks']['composite_score'],
            'qa_strength': results['call_strength']['qa_section']['composite_score'],
            'concern_ratio': results['analyst_concerns']['overall_concern_ratio'],
            'evasive_responses': results['analyst_concerns'].get('evasive_responses', 0),
            'uncertainty_indicators': results['analyst_concerns'].get('uncertainty_responses', 0),
            'question_count': results['question_count'],
            'word_count': results['word_count'],
            'eps_vs_expectations': 0,  # Default values for missing earnings data
            'guidance_vs_expectations': 0
        }
        
        # Add VADER scores if available
        if 'vader_compound' in results['overall_sentiment']:
            features['overall_vader'] = results['overall_sentiment']['vader_compound']
            features['prepared_vader'] = results['prepared_remarks_sentiment']['vader_compound']
            features['qa_vader'] = results['qa_sentiment']['vader_compound']
        
        # Make prediction using saved model
        prediction = dummy_analyzer.predict_stock_movement(saved_model, features)
        predictions.append({
            'file': file_key,
            'predicted_movement': prediction
        })
    
    # Return results in the same format as build_predictive_model
    return {
        'r2': saved_model.get('r2', 0),
        'mse': saved_model.get('mse', 0),
        'feature_importance': saved_model.get('feature_importance', []),
        'predictions_new': predictions  # Use different key to distinguish from training predictions
    }

def prepare_results_for_web(analysis_results, model_results=None, model_available=False):
    """Convert analysis results to web-friendly format"""
    web_results = []
    
    for file_key, results in analysis_results.items():
        # Calculate summary metrics
        overall_sentiment = results['overall_sentiment']['textblob_polarity']
        sentiment_label = get_sentiment_label(overall_sentiment)
        
        # Get top question categories
        top_categories = dict(results['question_categories'].most_common(3))
        
        # Calculate concern level
        concern_ratio = results['analyst_concerns']['overall_concern_ratio']
        concern_level = get_concern_level(concern_ratio)
        
        # Get call strength
        call_strength = results['call_strength']['overall_strength']
        strength_level = get_strength_level(call_strength)
        
        web_result = {
            'file_name': results['file_name'],
            'overall_sentiment': round(overall_sentiment, 3),
            'sentiment_label': sentiment_label,
            'sentiment_color': get_sentiment_color(overall_sentiment),
            'prepared_sentiment': round(results['prepared_remarks_sentiment']['textblob_polarity'], 3),
            'qa_sentiment': round(results['qa_sentiment']['textblob_polarity'], 3),
            'question_count': results['question_count'],
            'word_count': results['word_count'],
            'top_categories': top_categories,
            'concern_ratio': round(concern_ratio, 3),
            'concern_level': concern_level,
            'concern_color': get_concern_color(concern_ratio),
            'call_strength': round(call_strength, 2),
            'strength_level': strength_level,
            'strength_color': get_strength_color(call_strength),
            'questions': results['questions'][:10],  # Limit to first 10 questions for display
            'analyst_concerns': results['analyst_concerns']
        }
        
        # Add VADER scores if available
        if 'vader_compound' in results['overall_sentiment']:
            web_result['vader_compound'] = round(results['overall_sentiment']['vader_compound'], 3)
            web_result['prepared_vader'] = round(results['prepared_remarks_sentiment']['vader_compound'], 3)
            web_result['qa_vader'] = round(results['qa_sentiment']['vader_compound'], 3)
        
        web_results.append(web_result)
    
    # Prepare model results for display
    model_info = {
        'available': model_available,
        'results': None
    }
    
    if model_results:
        # Handle both training predictions and new predictions
        predictions_data = []
        if 'predictions' in model_results:  # Training mode
            predictions_data = model_results['predictions'].to_dict('records')
        elif 'predictions_new' in model_results:  # Using saved model
            predictions_data = model_results['predictions_new']
            
        # Check if feature importance values are all zero (invalid model)
        feature_importance_data = model_results['feature_importance'].head(5).to_dict('records') if hasattr(model_results['feature_importance'], 'head') else model_results['feature_importance'][:5]
        
        # Check if all importance values are zero
        has_valid_importance = False
        if feature_importance_data:
            has_valid_importance = any(item.get('importance', 0) > 0 for item in feature_importance_data)
        
        # If all importances are zero, create a warning message
        if not has_valid_importance:
            feature_importance_data = [
                {'feature': 'Model needs retraining', 'importance': 0.0},
                {'feature': 'Insufficient training data detected', 'importance': 0.0},
                {'feature': 'Please retrain with more diverse data', 'importance': 0.0},
                {'feature': 'Current predictions may be unreliable', 'importance': 0.0},
                {'feature': 'Go to Train Model page to fix this', 'importance': 0.0}
            ]
        
        model_info['results'] = {
            'r2_score': round(model_results['r2'], 3),
            'mse': round(model_results['mse'], 3),
            'feature_importance': feature_importance_data,
            'predictions': predictions_data,
            'using_saved_model': 'predictions_new' in model_results,
            'model_needs_retraining': not has_valid_importance
        }
    
    return {'transcripts': web_results, 'model': model_info}

def get_sentiment_label(sentiment):
    if sentiment > 0.1:
        return "Positive"
    elif sentiment < -0.1:
        return "Negative"
    else:
        return "Neutral"

def get_sentiment_color(sentiment):
    if sentiment > 0.1:
        return "success"
    elif sentiment < -0.1:
        return "danger"
    else:
        return "warning"

def get_concern_level(ratio):
    if ratio > 0.4:
        return "High"
    elif ratio > 0.2:
        return "Medium"
    else:
        return "Low"

def get_concern_color(ratio):
    if ratio > 0.4:
        return "danger"
    elif ratio > 0.2:
        return "warning"
    else:
        return "success"

def get_strength_level(strength):
    if strength > 3.0:
        return "Strong"
    elif strength > 1.5:
        return "Moderate"
    else:
        return "Weak"

def get_strength_color(strength):
    if strength > 3.0:
        return "success"
    elif strength > 1.5:
        return "warning"
    else:
        return "danger"

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    temp_dir = tempfile.mkdtemp()
    
    try:
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                uploaded_files.append(file_path)
        
        if not uploaded_files:
            return jsonify({'error': 'No valid PDF files provided'}), 400
        
        analyzer = EarningsTranscriptAnalyzer(temp_dir)
        analyzer.process_all_transcripts()
        
        results = prepare_results_for_web(analyzer.analysis_results)
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train predictive model with transcripts and earnings data"""
    if 'transcript_files' not in request.files or 'earnings_file' not in request.files:
        flash('Both transcript files and earnings data file are required for training')
        return redirect(url_for('train'))
    
    transcript_files = request.files.getlist('transcript_files')
    earnings_file = request.files['earnings_file']
    
    if not transcript_files or transcript_files[0].filename == '':
        flash('No transcript files selected')
        return redirect(url_for('train'))
    
    if not earnings_file or earnings_file.filename == '':
        flash('No earnings data file selected')
        return redirect(url_for('train'))
    
    # Validate file types
    for file in transcript_files:
        if not allowed_file(file.filename):
            flash(f'Invalid transcript file type: {file.filename}. Only PDF files are allowed.')
            return redirect(url_for('train'))
    
    if not allowed_excel_file(earnings_file.filename):
        flash(f'Invalid earnings file type: {earnings_file.filename}. Only Excel files (.xlsx, .xls) are allowed.')
        return redirect(url_for('train'))
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save transcript files
        transcript_paths = []
        for file in transcript_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                transcript_paths.append(file_path)
        
        # Save earnings file
        earnings_filename = secure_filename(earnings_file.filename)
        earnings_path = os.path.join(temp_dir, earnings_filename)
        earnings_file.save(earnings_path)
        
        # Process transcripts and train model
        analyzer = EarningsTranscriptAnalyzer(temp_dir, earnings_path)
        analyzer.process_all_transcripts()
        
        if not analyzer.analysis_results:
            flash('No analysis results generated. Please check your PDF files.')
            return redirect(url_for('train'))
        
        # Build predictive model
        model_results = analyzer.build_predictive_model()
        
        if not model_results:
            flash('Failed to build predictive model. Please check your earnings data format.')
            return redirect(url_for('train'))
        
        # Save the trained model
        if save_trained_model(model_results):
            flash('Model trained and saved successfully!', 'success')
        else:
            flash('Model trained but failed to save. You can still use it for this session.', 'warning')
        
        # Prepare training results for display
        training_results = prepare_training_results(analyzer.analysis_results, model_results, transcript_paths, earnings_path)
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return render_template('training_results.html', results=training_results)
        
    except Exception as e:
        flash(f'Error during training: {str(e)}')
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return redirect(url_for('train'))

def prepare_training_results(analysis_results, model_results, transcript_paths, earnings_path):
    """Prepare training results for display"""
    return {
        'model_performance': {
            'r2_score': round(model_results['r2'], 3),
            'mse': round(model_results['mse'], 3),
            'feature_importance': model_results['feature_importance'].head(10).to_dict('records'),
            'predictions': model_results['predictions'].to_dict('records') if 'predictions' in model_results else []
        },
        'training_data': {
            'transcript_count': len(analysis_results),
            'transcript_files': [os.path.basename(path) for path in transcript_paths],
            'earnings_file': os.path.basename(earnings_path),
            'data_points': len(model_results['predictions']) if 'predictions' in model_results else 0
        }
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)