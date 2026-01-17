# QTracker

## Overview

QTracker is a Python-based project designed for tracking and analyzing data efficiently. The codebase is structured to facilitate modular development, making it easy to extend and maintain. The main components of the project include:

- **Data Preprocessing**: Located in the `preprocessing` directory, this module handles data loading and preparation.
- **Modeling**: The `models` directory contains various models used for tracking, including multi-track finders and refiners.
- **Layers**: The `layers` directory includes custom layers such as axial attention, which are essential for the model's performance.
- **Scripts**: The `scripts` directory is for utility scripts that assist in running experiments or processing data.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd QTracker_main
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   ```bash
   # On Windows
   .venv\Scripts\activate
   ```
4. Install the required packages using `uv sync`:
   ```bash
   uv sync
   ```

## Development Practices

- **Pull Requests**: Always create a pull request (PR) for any changes instead of merging directly into the main branch. This practice ensures code review and maintains code quality.
- **Pre-commit Hooks**: To maintain code quality, use pre-commit hooks. Install them by running:
  ```bash
  pre-commit install
  ```
  This will set up hooks to automatically check your code for issues before committing.

## Running Tests

To run the tests, use the following command:

```bash
pytest
```

## Contributing

Contributions are welcome! Please follow the guidelines outlined in the repository for submitting issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
