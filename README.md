# OptiLang

**A Python-inspired interpreter with real-time code analysis and optimization suggestions**

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Project Overview

OptiLang is an educational interpreter for a Python-like language (PyLite) that provides:
- **Real-time code execution** with line-by-line profiling
- **Optimization suggestions** based on detected anti-patterns
- **Quantitative scoring** (0-100) for code quality
- **Pattern detection** for 8+ common performance issues

---

## 🚀 Quick Start

### Installation

```bash
pip install optilang
```

### Basic Usage

```python
from optilang import execute, analyze

# Execute PyLite code
result = execute("""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
""")

print(result.output)  # "120"
print(f"Execution time: {result.execution_time}ms")

# Analyze code for optimizations
report = analyze("""
for i in range(100):
    for j in range(100):
        result = i * j
""")

print(f"Optimization Score: {report.optimization_score}/100")
for suggestion in report.suggestions:
    print(f"Line {suggestion.line}: {suggestion.description}")
```

---

## 🏗️ Architecture

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Lexer   │ -> │  Parser  │ -> │ Executor │ -> │ Profiler │
│ (Tokens) │    │  (AST)   │    │ (Runtime)│    │ (Metrics)│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                       │              │
                                       v              v
                                 ┌──────────┐    ┌──────────┐
                                 │Optimizer │    │  Scorer  │
                                 │(Patterns)│    │ (0-100)  │
                                 └──────────┘    └──────────┘
```

---

## 📋 Features

### Current (v0.1.0)
- [x] Lexical analysis (tokenization)
- [x] Syntax parsing (AST generation)
- [x] Code execution (variables, functions, control flow)
- [x] Basic profiling (execution time, line counts)

### Planned
- [ ] Advanced profiling (memory usage, call graphs)
- [ ] 8+ optimization patterns
- [ ] ML-based suggestion ranking (optional)
- [ ] Optimization score calculation
- [ ] Comprehensive documentation

---

## 🛠️ Development Setup

```bash
# Clone repository
git clone https://github.com/Sthamanik/optilang.git
cd optilang

# Create virtual environment
conda create -n optilang python -y
conda activate optilang

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black optilang/

# Type checking
mypy optilang/

# Linting
flake8 optilang/
```

---

## 📚 Documentation

- **User Guide**: Coming soon
- **API Reference**: Coming soon
- **Contributing Guide**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md)

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=optilang --cov-report=html

# View coverage report
open htmlcov/index.html  # Linux/Mac
# or start htmlcov/index.html on Windows
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

- **Manik Kumar Shrestha** - [GitHub](https://github.com/Sthamanik)
- **Om Shree Mahat** - *Developer*
- **Aashish Rimal** - *Developer*

---

## 📧 Contact

For questions or feedback:
- **Email**: shresthamanik1820@gmail.com
- **Issues**: [GitHub Issues](https://github.com/Sthamanik/optilang/issues)

---

**⭐ Star this repository if you find it useful!**
