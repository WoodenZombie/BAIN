# BAIN Course - Unicorn University

Repository for BAIN (Artificial Intelligence) course hometasks at Unicorn University.

## Overview

This repository contains coursework assignments and projects for the BAIN course, managed as a Python project using Poetry for dependency management.

## Project Details

- **Course**: BAIN (Artificial Intelligence)
- **University**: Unicorn University
- **Author**: Tetiana Velehura
- **Python Version**: >=3.14
- **Package Manager**: Poetry

## Development

### Development Dependencies

The project includes the following development tools:

- **pre-commit** (>=4.5.1): Git hooks framework
- **ruff** (>=0.15.1): Fast Python linter and formatter
- **mypy** (>=1.15.0): Static type checker for Python

### Code Quality

The project enforces strict code quality standards:

- **Ruff Configuration**:
  - Line length: 88 characters
  - Rules enabled: E (errors), F (Pyflakes), I (imports), S (security)

- **Mypy Configuration**:
  - Strict mode enabled for comprehensive type checking

### Running Checks

To run pre-commit checks on all files:
```bash
pre-commit run --all-files
```

To check code with ruff:
```bash
ruff check .
```

To run type checking with mypy:
```bash
mypy .
```
