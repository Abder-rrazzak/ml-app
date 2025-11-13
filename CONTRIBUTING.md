# 🤝 Contributing to ML Iris Classification

Thank you for your interest in contributing to this MLOps project! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [🎯 How to Contribute](#-how-to-contribute)
- [🔧 Development Setup](#-development-setup)
- [📝 Code Standards](#-code-standards)
- [🧪 Testing Guidelines](#-testing-guidelines)
- [📚 Documentation](#-documentation)
- [🐛 Bug Reports](#-bug-reports)
- [✨ Feature Requests](#-feature-requests)
- [🔄 Pull Request Process](#-pull-request-process)

## 🎯 How to Contribute

We welcome contributions in several forms:

- 🐛 **Bug fixes**
- ✨ **New features**
- 📚 **Documentation improvements**
- 🧪 **Test coverage improvements**
- 🔧 **Performance optimizations**
- 📊 **MLOps pipeline enhancements**

## 🔧 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ml-app.git
cd ml-app

# Add upstream remote
git remote add upstream https://github.com/abder-rrazzak/ml-app.git
```

### 2. Environment Setup

```bash
# Complete development setup
make dev-setup

# Or manual setup
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev,docs,viz,mlops]"
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
make test

# Check code quality
make lint
make type-check
```

## 📝 Code Standards

### Python Code Style

- **Formatter**: Black (line length: 88)
- **Import sorting**: isort (Black-compatible profile)
- **Linting**: flake8 with extensions
- **Type checking**: mypy
- **Security**: bandit

### Code Quality Checklist

- ✅ All code formatted with Black
- ✅ Imports sorted with isort
- ✅ No linting errors (flake8)
- ✅ Type hints for all public functions
- ✅ Docstrings for all modules, classes, and functions
- ✅ No security vulnerabilities (bandit)
- ✅ Test coverage ≥ 90%

### Pre-commit Hooks

All commits are automatically checked for:

```bash
# Run all pre-commit hooks manually
make pre-commit

# Individual checks
make format    # Black + isort
make lint      # flake8
make type-check # mypy
make security  # bandit
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **API Tests**: Test REST endpoints
4. **Performance Tests**: Test response times and load

### Writing Tests

```python
# Example test structure
import pytest
from src.model import IrisClassifier

class TestIrisClassifier:
    """Test suite for IrisClassifier."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        classifier = IrisClassifier()
        assert not classifier.is_trained
    
    def test_training(self, sample_data):
        """Test model training."""
        classifier = IrisClassifier()
        X_train, y_train = sample_data
        
        classifier.train(X_train, y_train)
        assert classifier.is_trained
```

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_model.py -v

# Specific test function
pytest tests/test_model.py::TestIrisClassifier::test_training -v
```

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def train_model(X_train, y_train, random_state=42):
    """
    Train the Iris classification model.
    
    Args:
        X_train: Training features as pandas DataFrame or numpy array
        y_train: Training labels as pandas Series or numpy array
        random_state: Random seed for reproducibility
        
    Returns:
        IrisClassifier: Trained model instance
        
    Raises:
        ValueError: If training data is empty or invalid
        
    Example:
        >>> classifier = train_model(X_train, y_train)
        >>> predictions = classifier.predict(X_test)
    """
```

### Documentation Updates

- Update docstrings for any modified functions
- Update README.md for new features
- Add examples for new functionality
- Update API documentation if endpoints change

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Ensure you're using the latest version
3. Test with a minimal example

### Bug Report Template

```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With input '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 1.0.0]

**Additional Context**
Any other context about the problem.
```

## ✨ Feature Requests

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
Your ideas for how this could be implemented.

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other context or screenshots.
```

## 🔄 Pull Request Process

### 1. Create Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following our standards
- Add/update tests
- Update documentation
- Ensure all checks pass

### 3. Test Your Changes

```bash
# Run full test suite
make test-cov

# Check code quality
make pre-commit

# Test API if applicable
make api-test
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new feature description

- Detailed explanation of changes
- Any breaking changes
- Related issue numbers"
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Related issues linked

### PR Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Maintainers review code and provide feedback
3. **Testing**: Additional testing if needed
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to main branch

## 🏷️ Commit Message Convention

We follow conventional commits:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
feat(api): add batch prediction endpoint

- Support for multiple samples in single request
- Improved error handling for large batches
- Updated API documentation

Closes #123
```

## 🎖️ Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes for significant contributions
- GitHub contributors page

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private matters

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to make this project better! 🚀