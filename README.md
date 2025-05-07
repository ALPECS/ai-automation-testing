# Calculus Problem Tester using OpenRouter

This Streamlit application tests an AI model's ability to solve calculus problems presented as images. It uses OpenRouter to interact with a specified AI model (e.g., GPT-4o) and SymPy for evaluating the correctness of the solutions.

## Features

*   **Streamlit UI**: Provides a user-friendly interface to initiate test runs, view live logs, and see results.
*   **Problem Loading**: Loads calculus problems from a `problems.csv` file. Each entry includes an image name, input features, context features, and the correct answer.
*   **Image Processing**: Sends images of calculus problems (stored in the `calculus_problems/` directory) to an AI model.
*   **AI Interaction**: Leverages OpenRouter to communicate with various AI models.
*   **Response Evaluation**:
    *   Extracts the final answer from the AI's response.
    *   Compares the AI's answer with the correct answer using either:
        *   Simple string comparison.
        *   Mathematical equivalence checking via SymPy.
*   **Results Logging**: Saves detailed results of each test run (including the problem, AI's answer, and evaluation) to a CSV file (`results_openrouter.csv`).
*   **Downloadable Results**: Allows users to download the results CSV directly from the Streamlit interface.

## Project Structure

```
.
├── .gitignore
├── app.py                      # Main Streamlit application
├── calculus_problems/          # Directory containing problem images
│   └── image.png
├── chatgpt_testing.py          # Core script for testing logic
├── problems.csv                # CSV file defining the problems and answers
├── requirements.txt            # Python dependencies
├── results_openrouter.csv      # Output CSV for test results (created after a run)
└── README.md                   # This file
```

## How it Works

1.  **Setup**:
    *   Problem definitions (image name, features, correct answer) are listed in `problems.csv`.
    *   Corresponding images are placed in the `calculus_problems/` directory.
    *   An OpenRouter API key needs to be set as an environment variable (`OPENROUTER_API_KEY`) in a `.env` file.
2.  **Execution (`app.py`)**:
    *   The Streamlit app (`app.py`) provides a button to "Start Testing".
    *   Clicking the button triggers the `run_test()` function in `chatgpt_testing.py`.
3.  **Testing Process (`chatgpt_testing.py`)**:
    *   `load_problems()`: Reads `problems.csv`.
    *   For each problem:
        *   The image is encoded to base64.
        *   `get_openrouter_response()`: The image and a prompt are sent to the configured AI model via OpenRouter. The prompt instructs the AI to provide its final answer in a format suitable for SymPy.
        *   `evaluate_answer()`: The AI's extracted answer is compared to the `correct_answer` from the CSV. Evaluation can be a simple string match or a SymPy-based mathematical comparison.
        *   Results (problem details, AI answer, evaluation outcome) are collected.
    *   `save_results_to_csv()`: All results are saved to `results_openrouter.csv`.
4.  **Display**:
    *   The Streamlit app displays logs in real-time.
    *   Upon completion, the test results are shown in a table and can be downloaded as a CSV.

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
        *   `input_feature`: Descriptive feature of the input.
        *   `context_feature`: Descriptive context for the problem.
        *   `correct_answer`: The correct solution, formatted as a string that SymPy can parse (e.g., `'C * exp(-2*x) - exp(-3*x)'`).
6.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
7.  Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
8.  Click the "▶️ Start Testing" button to begin.

## Key Files

*   `app.py`: The main Streamlit web application.
*   `chatgpt_testing.py`: Contains the core logic for loading problems, interacting with the OpenRouter API, and evaluating answers.
*   `problems.csv`: Defines the calculus problems, their corresponding images, and correct answers.
*   `calculus_problems/`: Directory to store the image files for the calculus problems.
*   `requirements.txt`: Lists the Python packages required to run the application.
*   `.env` (create this): Used to store the `OPENROUTER_API_KEY`.
*   `results_openrouter.csv`: Stores the output of the test runs (created automatically).

## Dependencies

The main dependencies are listed in `requirements.txt` and include:
*   streamlit
*   pandas
*   openai
*   python-dotenv
*   Pillow
*   sympy

Refer to `requirements.txt` for specific versions. 