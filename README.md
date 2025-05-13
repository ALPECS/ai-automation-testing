# Calculus Problem AI Tester and Results Dashboard

This project provides tools to test an AI model's ability to solve calculus problems presented as images and to visualize the aggregated test results using a Dash dashboard. It uses OpenRouter to interact with specified AI models and SymPy for evaluating the correctness of mathematical solutions.

## Features

*   **AI Model Testing (`ai_testing.py`):**
    *   Loads calculus problems from a `problems.csv` file (image name, features, correct answer).
    *   Sends images from `calculus_problems/` to an AI model via OpenRouter.
    *   Extracts and evaluates the AI's final answer against the correct answer (string comparison or SymPy-based mathematical equivalence).
    *   Saves detailed results to `aggregated_results_pydantic.csv`.
*   **Results Analysis Dashboard (`dashboard.py`):**
    *   Provides a web-based interface built with Dash and Plotly.
    *   Displays overall statistics: total unique problems, total models evaluated.
    *   Shows per-model performance at a glance: problems attempted, total runs, accuracy.
    *   Visualizes overall evaluation outcome distribution by model using bar charts (showing counts and percentages).
    *   Offers per-problem analysis with evaluation outcomes by model (bar charts with counts and percentages).
    *   Uses Bootstrap for a clean and responsive layout.
*   **Downloadable Results**: The underlying testing script likely allows for results to be saved, and the dashboard reads from this generated CSV.

## Project Structure

```
.
├── .gitignore
├── ai_testing.py      # Core script for testing logic & generating aggregated_results_pydantic.csv (inferred name)
├── dashboard.py                # Dash application for visualizing aggregated results
├── calculus_problems/          # Directory containing problem images
│   └── image.png
├── problems.csv                # CSV file defining the problems and answers
├── requirements.txt            # Python dependencies
├── aggregated_results_pydantic.csv # Output CSV from ai_testing_pydantic.py, input for dashboard.py (created after a run)
└── README.md                   # This file
```

## How It Works

1.  **Setup**:
    *   Problem definitions (image name, features, correct answer) are listed in `problems.csv`.
    *   Corresponding images are placed in the `calculus_problems/` directory.
    *   An OpenRouter API key needs to be set as an environment variable (`OPENROUTER_API_KEY`) in a `.env` file.
2.  **Test Execution (`ai_testing_pydantic.py`)**:
    *   Run the `ai_testing_pydantic.py` script.
    *   This script loads problems, interacts with the AI model via OpenRouter for each problem, evaluates the answers, and saves the collated results into `aggregated_results_pydantic.csv`.
3.  **Results Visualization (`dashboard.py`)**:
    *   Run the `dashboard.py` script.
    *   This Dash application loads data from `aggregated_results_pydantic.csv`.
    *   It then presents various charts and statistics in your web browser, allowing for analysis of model performance.

## Setup and Usage

1.  **Clone the repository (if applicable).**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes `dash`, `pandas`, `plotly`, `dash-bootstrap-components`, `openai`, `python-dotenv`, `Pillow`, `sympy` and other necessary packages for `ai_testing_pydantic.py`)*
4.  **Configure API Key:**
    *   Create a `.env` file in the root directory.
    *   Add your OpenRouter API key to it:
        ```
        OPENROUTER_API_KEY="your_openrouter_api_key_here"
        ```
5.  **Prepare Problems:**
    *   Add your problem images (e.g., `.png`) to the `calculus_problems/` directory.
    *   Update `problems.csv` with the details for each problem:
        *   `image_name`: Filename of the image in `calculus_problems/`.
        *   `input_feature`: Descriptive feature of the input (if used by testing script).
        *   `context_feature`: Descriptive context for the problem (if used by testing script).
        *   `correct_answer`: The correct solution, formatted as a string that SymPy can parse (e.g., `'C * exp(-2*x) - exp(-3*x)'`).
6.  **Run the AI testing script:**
    ```bash
    python ai_testing.py
    ```
    *(This will generate/update the `aggregated_results_pydantic.csv` file)*
7.  **Run the Dashboard application:**
    ```bash
    python dashboard.py
    ```
8.  Open your web browser and navigate to the URL provided by Dash (usually `http://0.0.0.0:8050` or `http://127.0.0.1:8050`).

## Dependencies

The main dependencies are typically listed in `requirements.txt`. Key packages for this workflow include:
*   `dash`
*   `dash-bootstrap-components`
*   `plotly`
*   `pandas`
*   `openai` (or a similar library for OpenRouter interaction)
*   `python-dotenv`
*   `Pillow`
*   `sympy`

Refer to `requirements.txt` for specific versions and a complete list.
