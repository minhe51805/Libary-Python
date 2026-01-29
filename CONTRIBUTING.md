# Contributing to scanLt

First off, thank you for considering contributing to scanLt! It's people like you that make open source projects such as this a success.

## How Can I Contribute?

### Reporting Bugs

- **Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/your-org/scanLt/issues).
- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/your-org/scanLt/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements

- Open a new issue, clearly describing the proposed enhancement and its potential benefits. Provide as much detail and context as possible.

### Pull Request Process

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes.
4. Make sure your code lints.
5. Issue that pull request!

## Development Setup

1. Fork the `scanLt` repo on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/scanLt.git
   ```
3. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
4. Install the project in editable mode with all development dependencies:
   ```bash
   pip install -e ".[torch,onnx,mediapipe]"
   ```

Thank you for your contribution!
