# Earnings Transcript Analyzer

A powerful AI-powered web application that analyzes earnings call transcripts to extract sentiment, categorize questions, assess call strength, detect analyst concerns, and predict stock movements using machine learning.

![Earnings Analyzer](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

### 📊 **Comprehensive Analysis**
- **Sentiment Analysis**: TextBlob & VADER algorithms analyze emotional tone
- **Question Categorization**: AI categorizes analyst questions into 8 topics
- **Call Strength Assessment**: Composite scoring of confidence and uncertainty levels
- **Concern Detection**: Pattern recognition identifies evasive language and red flags

### 🤖 **Predictive Modeling**
- **Stock Movement Predictions**: Random Forest ML model predicts price movements
- **Feature Importance**: Shows which factors most influence predictions
- **Model Persistence**: Trained models are saved and reused across sessions
- **Training Interface**: Easy-to-use training workflow with drag & drop

### 🌐 **Modern Web Interface**
- **Drag & Drop Upload**: Upload multiple PDF transcripts simultaneously
- **Real-time Analysis**: Process and display results instantly
- **Interactive Dashboards**: Beautiful visualizations with detailed explanations
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/earnings-analyzer.git
   cd earnings-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://localhost:5000
   ```

## 🚀 Deploy to Railway (Recommended Cloud Hosting)

Deploy your own live version of this app to Railway in minutes:

### **One-Click Deploy**
1. **Fork this repository** to your GitHub account
2. **Sign up at [railway.app](https://railway.app)** (free tier available)
3. **Click "New Project"** → **"Deploy from GitHub repo"**
4. **Select your forked repository**
5. **Railway automatically detects and deploys** your Python Flask app
6. **Get your live URL** (e.g., `https://your-app-name.up.railway.app`)

### **Why Railway?**
- ✅ **Zero configuration** - Works with your existing code
- ✅ **File uploads persist** - PDF uploads and trained models are saved
- ✅ **Free tier** - Perfect for testing and light usage
- ✅ **Automatic scaling** - Handles traffic spikes seamlessly
- ✅ **Custom domains** - Add your own domain name
- ✅ **HTTPS included** - Secure connections out of the box

### **Railway Features**
- **Persistent Storage**: Uploaded files and ML models remain available
- **Environment Variables**: Configure settings without code changes
- **Automatic Deployments**: Push to GitHub → Automatic Railway deployment
- **Usage Metrics**: Monitor app performance and resource usage
- **Team Collaboration**: Share access with team members

**Live Demo**: *[Add your Railway URL here after deployment]*

## 📁 Project Structure

```
earnings-analyzer/
├── app.py                     # Flask web application
├── earnings_analyzer.py       # Core analysis engine
├── requirements.txt           # Python dependencies
├── Procfile                   # Railway deployment configuration
├── CLAUDE.md                 # Development guide
├── README.md                 # This file
├── templates/                # HTML templates
│   ├── base.html            # Base template
│   ├── index.html           # Upload page
│   ├── train.html           # Model training page
│   ├── results.html         # Analysis results
│   └── training_results.html # Training results
├── static/css/              # Stylesheets
│   └── style.css           # Custom styles
└── trained_models/          # Saved ML models (created automatically)
```

## 💡 How to Use

### 1. **Analyze Transcripts**
- Navigate to the main page
- Drag & drop PDF transcript files or click to browse
- View comprehensive analysis results with interactive explanations

### 2. **Train Predictive Model**
- Go to "Train Model" page
- Upload multiple PDF transcripts (3+ recommended)
- Upload Excel file with earnings data (see format below)
- Review training results and model performance

### 3. **File Format Requirements**

**PDF Transcript Naming:** Must follow `[TICKER]4Q24-Earnings-Call-Transcript.pdf` format
- Examples: `CPB4Q24-Earnings-Call-Transcript.pdf`, `4Q24-Earnings-Call-Transcript.pdf`
- Quarter must match Excel file exactly

**Excel Format for Training:** Supports 2-5 columns
```
Earnings | EPS_vs_Expectations | Guidance_vs_Expectations | Stock_Reaction
4Q24     | Beat               | Meet                    | +4.5
1Q25     | Miss               | Beat                    | -2.1
```

- **Earnings Column**: Must use format `4Q24`, `1Q25`, `2Q25`, `3Q25` (matches transcript filename)
- **Optional Columns**: EPS vs Expectations, Guidance vs Expectations ("Beat", "Miss", "Meet")
- **Stock_Reaction**: Stock price movement (% change, e.g., "+5.2", "-3.1")

## 🔧 Technical Details

### Core Technologies
- **Backend**: Flask (Python web framework)
- **ML/AI**: scikit-learn, TextBlob, VADER Sentiment
- **Data Processing**: pandas, numpy, pdfplumber
- **Frontend**: Bootstrap 5, JavaScript, CSS3
- **Model Storage**: pickle, joblib

### Analysis Components

#### **Sentiment Analysis**
- **Range**: -1.0 to +1.0
- **Algorithm**: TextBlob polarity scoring
- **Interpretation**: Positive (+0.1 to +1.0), Neutral (-0.1 to +0.1), Negative (-1.0 to -0.1)

#### **Call Strength**
- **Range**: 0 to 8+ (typically 0-6)
- **Method**: Composite scoring of confidence vs uncertainty indicators
- **Factors**: Financial strength terms, confidence words, uncertainty expressions

#### **Concern Level**
- **Range**: 0% to 100%
- **Method**: Percentage of responses containing concern indicators
- **Patterns**: Hedging language, uncertainty, defensive responses, evasive answers

#### **Predictive Model**
- **Algorithm**: Random Forest Regressor
- **Features**: Sentiment scores, call strength, concern ratios, earnings performance
- **Output**: Stock price movement predictions with confidence intervals

## 📊 Score Interpretations

### Overall Sentiment (-1.0 to +1.0)
- **+0.5 to +1.0**: Very confident, optimistic tone
- **+0.1 to +0.5**: Moderately positive, cautious optimism
- **-0.1 to +0.1**: Neutral, matter-of-fact presentation
- **-0.5 to -0.1**: Somewhat negative, concerns discussed
- **-1.0 to -0.5**: Very negative, significant problems

### Call Strength (0 to 8+)
- **4.0+**: Very strong call, high confidence
- **3.0-4.0**: Strong call, good confidence
- **2.0-3.0**: Moderate strength, balanced
- **1.0-2.0**: Weak call, more uncertainty
- **0-1.0**: Very weak call, high uncertainty

### Concern Level (0% to 100%)
- **0-10%**: Excellent, direct responses
- **10-20%**: Good, mostly straightforward
- **20-30%**: Moderate concern, some defensive language
- **30-40%**: Elevated concern, frequent hedging
- **40%+**: High concern, significant uncertainty

## 🛠️ Development

### Setting up for Development
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run in debug mode: `python app.py`

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit your changes: `git commit -am 'Add new feature'`
6. Push to the branch: `git push origin feature-name`
7. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/YOUR_USERNAME/earnings-analyzer/issues)
- **Documentation**: See [CLAUDE.md](CLAUDE.md) for detailed development guide
- **Contact**: [Your Contact Information]

## 🏆 Acknowledgments

- Built with Flask and scikit-learn
- Sentiment analysis powered by TextBlob and VADER
- PDF processing using pdfplumber
- UI components from Bootstrap 5
- Icons from Font Awesome

---

**Note**: This tool is for educational and research purposes. Investment decisions should not be based solely on automated analysis results.