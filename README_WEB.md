# Earnings Transcript Analyzer - Web Interface

A beautiful web-based interface for analyzing earnings call transcripts using AI-powered sentiment analysis, question categorization, and concern detection.

## Features

- **Drag & Drop Upload**: Easily upload one or more PDF transcript files
- **Real-time Analysis**: AI-powered analysis of sentiment, questions, and concerns
- **Beautiful Results**: Clean, interactive dashboard with detailed insights
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Export Functionality**: Download results as CSV/Excel files

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Web Application**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5000` in your web browser

## How to Use

1. **Upload Transcripts**: 
   - Drag and drop PDF files onto the upload area, or
   - Click "Browse Files" to select files manually
   - Multiple files can be uploaded at once

2. **View Results**:
   - Overall sentiment analysis with visual indicators
   - Call strength assessment
   - Analyst concern detection
   - Detailed question categorization
   - Interactive tabs for different analysis aspects

3. **View Results**:
   - All results are displayed directly in the browser
   - No files are generated or saved to disk

## Web Interface Components

### Upload Page (`/`)
- Drag & drop file upload area
- File validation (PDF only)
- Progress indicators
- Feature overview

### Results Page (`/results`)
- Key metrics dashboard
- Sentiment analysis with progress bars
- Question categorization charts
- Concern level indicators
- Detailed question list with sentiment scores

### API Endpoint (`/api/analyze`)
- RESTful API for programmatic access
- JSON response format
- Same analysis capabilities as web interface

## Technical Details

- **Framework**: Flask (Python web framework)
- **Frontend**: Bootstrap 5 + Custom CSS
- **File Handling**: Secure file upload with validation
- **Analysis Engine**: Same core `EarningsTranscriptAnalyzer` class
- **Responsive Design**: Mobile-friendly interface

## File Structure

```
├── app.py                 # Flask web application
├── earnings_analyzer.py   # Core analysis engine
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Upload page
│   └── results.html      # Results page
└── static/css/           # Stylesheets
    └── style.css         # Custom styles
```

## Customization

- **Styling**: Modify `static/css/style.css` for custom themes
- **Templates**: Edit HTML templates in `templates/` folder
- **Analysis**: Extend `earnings_analyzer.py` for additional features
- **Configuration**: Update Flask settings in `app.py`

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Security Features

- File type validation (PDF only)
- Secure filename handling
- Temporary file cleanup
- CSRF protection ready
- File size limits (50MB max)