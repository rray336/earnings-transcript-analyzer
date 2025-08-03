# Contributing to Earnings Transcript Analyzer

We welcome contributions to the Earnings Transcript Analyzer! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub Issues tab to report bugs or request features
- Provide detailed information about the issue
- Include steps to reproduce the problem
- Mention your operating system and Python version

### Submitting Changes

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/earnings-analyzer.git
   cd earnings-analyzer
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test Your Changes**
   ```bash
   python app.py
   # Test the web interface thoroughly
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to GitHub and create a pull request
   - Provide a clear description of your changes
   - Reference any related issues

## 🎯 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Keep functions focused and small
- Add docstrings for new functions and classes

### Frontend Guidelines
- Use Bootstrap 5 classes consistently
- Maintain responsive design principles
- Follow the existing color scheme
- Test on multiple screen sizes

### Adding New Features

#### New Analysis Algorithms
1. Add the algorithm to `earnings_analyzer.py`
2. Update the web interface in `app.py`
3. Add corresponding HTML templates
4. Update the documentation

#### New ML Models
1. Implement in `earnings_analyzer.py`
2. Add training interface if needed
3. Update model persistence logic
4. Add performance metrics

#### UI Improvements
1. Maintain consistency with existing design
2. Test accessibility features
3. Ensure mobile responsiveness
4. Update CSS classes appropriately

## 📁 Project Structure

```
earnings-analyzer/
├── app.py                    # Main Flask application
├── earnings_analyzer.py      # Core analysis logic
├── templates/               # HTML templates
├── static/css/             # Stylesheets
├── requirements.txt        # Dependencies
└── README.md              # Project documentation
```

## 🧪 Testing

### Manual Testing Checklist
- [ ] Upload single PDF file
- [ ] Upload multiple PDF files
- [ ] Train model with valid data
- [ ] Test with invalid file formats
- [ ] Check responsive design on mobile
- [ ] Verify all modal explanations work
- [ ] Test model persistence across sessions

### Adding Tests
- Create test files in a `tests/` directory
- Use pytest for testing framework
- Include both unit tests and integration tests
- Test edge cases and error conditions

## 📝 Documentation

### Code Documentation
- Add docstrings to all public functions
- Include parameter descriptions and return types
- Provide usage examples for complex functions

### User Documentation
- Update README.md for new features
- Add screenshots for UI changes
- Update the Excel format requirements if changed
- Keep CLAUDE.md updated for developers

## 🐛 Bug Reports

Include the following information:
- Operating system and version
- Python version
- Browser (for web interface issues)
- Steps to reproduce
- Expected vs actual behavior
- Console errors (if any)
- Sample files (if applicable, anonymized)

## ✨ Feature Requests

When requesting features:
- Describe the use case clearly
- Explain the expected behavior
- Consider implementation complexity
- Suggest UI/UX improvements if applicable

## 🔄 Pull Request Process

1. **Before Submitting**
   - Test your changes thoroughly
   - Update documentation
   - Check code style
   - Ensure no breaking changes

2. **PR Description Should Include**
   - Clear description of changes
   - Screenshots for UI changes
   - Testing performed
   - Any breaking changes

3. **Review Process**
   - Maintainers will review your PR
   - Address any feedback
   - Keep PR focused and atomic

## 🚀 Development Setup

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/earnings-analyzer.git
cd earnings-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run in development mode
python app.py
```

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙋‍♂️ Questions?

Feel free to open an issue for any questions about contributing!

---

Thank you for contributing to the Earnings Transcript Analyzer! 🎉