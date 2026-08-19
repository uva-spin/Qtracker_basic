# QTracker

## Overview

QTracker is a Python-based project designed for tracking and analyzing data efficiently. The codebase is structured to facilitate modular development, making it easy to extend and maintain. The main components of the project include:

- **Data Preprocessing**: Located in the `preprocessing` directory, this module handles data loading and preparation.
- **Modeling**: The `models` directory contains various models used for tracking, including multi-track finders and refiners.
- **Layers**: The `layers` directory includes custom layers such as axial attention, which are essential for the model's performance.
- **Scripts**: The `scripts` directory is for utility scripts that assist in running experiments or processing data.

## Installation

To set up the project, follow these steps:

1. Clone the repository in bash (Mac) or WSL (Windows):
   ```bash
   git clone <repository-url>
   cd QTracker_main
   ```
2. Install [Anaconda](https://www.anaconda.com/download) and verify:
   ```bash
   conda --version
   ```
3. Create a new Anaconda environment with necessary dependencies:
   ```bash
   conda env create -f environment.yml
   ```
4. Activate Anaconda environment:
   ```bash
   conda activate qtracker
   ```

## Development Practices

- **Pull Requests**: Always create a pull request (PR) for any changes instead of merging directly into the main branch. This practice ensures code review and maintains code quality.
- **Pre-commit Hooks**: To maintain code quality, use pre-commit hooks. Install them by running:
  ```bash
  pre-commit install
  ```
  This will set up hooks to automatically check your code for issues before committing.

## Contributing

Contributions are welcome! Please follow the guidelines outlined in the repository for submitting issues and pull requests.
