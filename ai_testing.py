import os
import base64
import csv
import json # Added for JSON parsing
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from sympy import sympify, SympifyError # We'll uncomment and use this later for robust comparison
import logging
from pydantic import BaseModel, validator # Added Pydantic imports
import concurrent.futures # Added for parallel execution

# --- Pydantic Model Definition ---
class LLMResponse(BaseModel):
    derivation: str
    final_answer: str

    @validator('final_answer')
    def validate_final_answer(cls, value):
        if value == "Unable to solve":
            return value
        try:
            sympify(value) # Try to parse with SymPy
            return value
        except SympifyError:
            raise ValueError("Final answer must be a valid SymPy expression or 'Unable to solve'")
# --- End Pydantic Model Definition ---

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
# --- MODIFICATION: Define a list of models to test ---
AI_MODELS = ["openai/gpt-4o-mini", "google/gemini-2.0-flash-001"] # Added deepseek model

RESULTS_CSV_PATH = "results_pydantic.csv" # Path for individual run results (if saved manually)
AGGREGATED_RESULTS_CSV_PATH = "aggregated_results_pydantic.csv" # Path for aggregated results
NUM_PARALLEL_RUNS = 5 # Number of times to run the test in parallel
IMAGE_DIR = "calculus_problems"
PROBLEMS_CSV_PATH = "problems.csv"
# Configure logging to output to console by default. Streamlit app will add a handler.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not openrouter_api_key:
    logger.error("OPENROUTER_API_KEY not found in environment variables. Please set it in a .env file.")
    raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in a .env file.")

client = OpenAI(
    base_url=OPENROUTER_API_BASE,
    api_key=openrouter_api_key,
)

def load_problems(csv_path):
    """Loads problem data from the specified CSV file."""
    try:
        df = pd.read_csv(csv_path)

        # --- ADD: Strip whitespace from column names ---
        df.columns = df.columns.str.strip()

        # --- ADD: Strip leading/trailing whitespace from all string columns/values ---
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()

        # --- ADD: Strip surrounding quotes from correct_answer ---
        if 'correct_answer' in df.columns:
             def strip_quotes(s):
                 if isinstance(s, str) and len(s) > 1:
                     if (s.startswith("'") and s.endswith("'")) or \
                        (s.startswith('"') and s.endswith('"')):
                         return s[1:-1]
                 return s
             df['correct_answer'] = df['correct_answer'].apply(strip_quotes).str.strip() # Apply strip again after removing quotes

        # --- Remove fully blank rows (optional, but good practice) ---
        df.dropna(how='all', inplace=True)

        logger.info(f"Loaded {len(df)} problems from {csv_path}")
        # Basic validation
        # logger.info(df.head()) # Keep this for debugging if needed, or remove
        if 'image_name' not in df.columns or 'input_feature' not in df.columns or 'context_feature' not in df.columns or 'correct_answer' not in df.columns:
            logger.error("CSV must contain 'image_name', 'input_feature', 'context_feature', and 'correct_answer' columns.")
            raise ValueError("CSV must contain 'image_name', 'input_feature', 'context_feature', and 'correct_answer' columns.")
        return df
    except FileNotFoundError:
        logger.error(f"Error: Problems file not found at {csv_path}")
        return pd.DataFrame() # Return empty DataFrame
    except Exception as e:
        logger.error(f"Error loading problems: {e}")
        return pd.DataFrame()

# Function to encode the image
def encode_image_to_base64(image_path):
    """Encodes an image file to base64 format."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logger.warning(f"Image file not found at {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

# Function to call OpenRouter API
def get_openrouter_response(base64_image, model="mistralai/mistral-7b-instruct", max_tokens=1000):
    """
    Sends the image and prompt to the OpenRouter API, expects a JSON response,
    validates it, and returns the final answer.
    """

    system_rule = (
        "You are an expert calculus solver. Your task is to solve the problem provided. "
        "You MUST provide your response as a JSON object with two keys: "
        "1. \"derivation\": A string containing your detailed step-by-step derivation. "
        "2. \"final_answer\": A string representing the mathematical answer, formatted for SymPy. "
        "This means it should be a Python-parsable string, like '2*x**2 + sin(x)'. "
        "If the final answer is a floating-point number, round it to 5 decimal places. "
        "Use Python math syntax: `**` for power, `*` for multiplication, and standard function names "
        "like `sqrt()`, `log()`, `sin()`, `cos()`, `tan()`. "
        "Do NOT use LaTeX or any other formatting for this final_answer string. "
        "If you cannot solve the problem, the value for 'final_answer' MUST be exactly: \"Unable to solve\"."
        "Ensure your entire response is a single, valid JSON object."
    )

    user_prompt_text = (
        "Please solve the calculus problem shown in the image. "
        "Provide your response as a JSON object with two string fields: 'derivation' and 'final_answer'. "
        "The 'derivation' field should contain all your reasoning and work step-by-step. "
        "The 'final_answer' field must be the final simplified mathematical *expression*, formatted as a Python string "
        "suitable for SymPy's `sympify()` function (e.g., '(x+1)/2'). "
        "If the problem involves solving an equation for a variable (e.g., solving for 'y' in terms of 'x'), "
        "the 'final_answer' should be the expression that this variable equals. For example, if the solution is "
        "'y = x**2 + C', the 'final_answer' should be 'x**2 + C'. Do *not* include the 'y =' part or the full equation "
        "in the 'final_answer'. "
        "If the final answer is a floating-point number, please round it to 5 decimal places. "
        "If the solution is an implicit equation from which the variable cannot be expressed as a simple expression, "
        "or if you are unable to find a solution for any other reason, the 'final_answer' field must be exactly \"Unable to solve\". "
        "Your entire output should be only this JSON object."
    )

    if not base64_image:
        logger.error("Invalid image data passed to get_openrouter_response.")
        return "Error: Invalid image data. No image provided."

    messages = [
        {"role": "system", "content": system_rule},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens, # Consider increasing if derivations are long
            temperature=0,
            response_format={"type": "json_object"} # Request JSON output from supported models
        )
        raw_response_content = response.choices[0].message.content.strip()
        # logger.info(f"Raw LLM JSON response: {raw_response_content}")

        try:
            # Validate the response using the Pydantic model
            validated_response = LLMResponse.model_validate_json(raw_response_content)
            logger.info(f"Validated LLM Answer: {validated_response.final_answer}")
            # The entire validated_response.derivation could be logged or stored if needed
            return validated_response.final_answer
        except ValueError as pydantic_error: # Catches Pydantic validation errors
            logger.error(f"Pydantic validation error: {pydantic_error}. Raw response: {raw_response_content}")
            return f"Error - LLM returned an invalid response format"
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON decoding error: {json_error}. Raw response: {raw_response_content}")
            return f"Error: LLM response is not valid JSON - {json_error}"

    except Exception as e:
        logger.error(f"Error calling OpenRouter API: {e}")
        return f"Error: API call failed - {e}"

# Function to evaluate the answer (No changes needed here, but kept for completeness)
def evaluate_answer(llm_answer, correct_answer, simple_mode=True):
    """
    Evaluates if the LLM's answer matches the expected answer.

    Args:
        llm_answer (str): The answer provided by the LLM (e.g., via OpenRouter)
        correct_answer (str): The correct answer from the CSV
        simple_mode (bool): If True, does simple string comparison. If False, uses sympy for math comparison.

    Returns:
        str: "Correct", "Incorrect", "Likely Correct", or "Evaluation Error"
    """
    if not llm_answer or not correct_answer:
        return "Evaluation Error"

    # Remove error messages if present
    if llm_answer.startswith("Error:"):
        return "Evaluation Error"

    if correct_answer == "Unable to solve" and llm_answer == "Unable to solve":
        return "Correct, input is unsolvable!"
    
    # If llm answer is unable to solve, and the correct answer is solvable, return that it is incorrect
    if llm_answer == "Unable to solve" and correct_answer not in "Unable to solve":
        return "Incorrect, LLM is unable to solve, but the correct answer exists!"
    
    if correct_answer == "Unable to solve" and llm_answer != "Unable to solve":
        return "Incorrect, LLM is able to solve, but the correct answer is unsolvable!"
    
    if simple_mode:
        # Simple string comparison (basic)
        # Remove spaces and convert to lowercase for more flexible comparison
        processed_llm = llm_answer.replace(" ", "").lower()
        processed_correct = str(correct_answer).replace(" ", "").lower() # Ensure correct_answer is string

        # Direct comparison
        if processed_llm == processed_correct:
            return "Correct"

        # Check if correct answer is contained in LLM's answer
        # (Since LLM might explain the answer with additional text)
        # Be cautious with this, it might lead to false positives if the correct answer is short (e.g., "1")
        if processed_correct in processed_llm:
             # Add a check for length to avoid trivial matches
            if len(processed_correct) > 2: # Only consider "Likely Correct" if the answer is reasonably long
                return "Likely Correct"
            elif processed_llm == processed_correct: # If it's short, require exact match
                 return "Correct"


        return "Incorrect"
    else:
        # Attempt to treat answers as floats and compare with rounding
        try:
            llm_float = float(llm_answer)
            # Correct answer might be like "Float(\\"123.456\\")"
            # Try to extract the number part for correct_answer if it's a string containing "Float"
            correct_answer_str = str(correct_answer)
            if "Float(" in correct_answer_str and correct_answer_str.endswith(")"):
                 # Attempt to extract the numeric part within Float("...")
                 # This is a basic extraction, might need to be more robust
                 try:
                     # Example: Float("123.45") -> 123.45
                     # Example: Float('1.23e+5') -> 1.23e+5
                     # Need to handle quotes around the number carefully
                     start = correct_answer_str.find("(") + 1
                     end = correct_answer_str.rfind(")")
                     num_str = correct_answer_str[start:end]
                     # Remove potential inner quotes like Float(" '123.45' ")
                     num_str = num_str.strip().strip("'\"")
                     correct_float = float(num_str)
                 except ValueError:
                     # If extraction fails, fall back to original correct_answer for sympify
                     pass # correct_float will not be defined, sympify path will be taken
            else:
                correct_float = float(correct_answer_str)

            # If both are successfully converted to float
            if 'correct_float' in locals() and isinstance(llm_float, float) and isinstance(correct_float, float):
                # Round to 5 decimal places (or any desired precision)
                if round(llm_float, 5) == round(correct_float, 5):
                    return "Correct, Floats are equivalent (rounded)!"
                else:
                    return f"Incorrect, Answers are NOT mathematically equivalent!"
        except (ValueError, TypeError):
            # If conversion to float fails, proceed to SymPy comparison
            pass

        try:
            try:
                expr_llm = sympify(llm_answer)
                expr_correct = sympify(correct_answer)
                # First, try the .equals() method
                if expr_llm.equals(expr_correct):
                     return "Correct, Answers are mathematically equivalent! (via .equals())"
                # If .equals() is False, try simplifying the difference
                elif (expr_llm - expr_correct).simplify() == 0:
                    return "Correct, Answers are mathematically equivalent! (via simplified difference)"
                else:
                     # For debugging, log the simplified difference if it's not zero
                     logger.info(f"Simplified difference for non-equivalence: {(expr_llm - expr_correct).simplify()}")
                     return "Incorrect, Answers are NOT mathematically equivalent!"
            except (SympifyError, TypeError, SyntaxError) as symp_err:
                 logger.warning(f"SymPy Error comparing '{llm_answer}' and '{correct_answer}': {symp_err}")
                 return "Evaluation Error (SymPy Parse)"
        except Exception as e:
            logger.error(f"Error in sympy evaluation: {e}")
            return "Evaluation Error"

def save_results_to_csv(results_data, output_path):
    """Saves the results to a CSV file."""
    try:
        # Convert list of dictionaries or DataFrame to DataFrame
        if isinstance(results_data, list):
            results_df = pd.DataFrame(results_data)
        elif isinstance(results_data, pd.DataFrame):
            results_df = results_data # Already a DataFrame
        else:
            logger.error("Invalid data type for saving to CSV. Must be list or DataFrame.")
            return False

        # Save to CSV
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

# --- CHANGE: Renamed main to run_test and return results ---
def run_test():
    """Runs the full testing process and returns results as a DataFrame."""
    logger.info("Starting a Calculus Problem Test Run (using OpenRouter)...")
    problems_df = load_problems(PROBLEMS_CSV_PATH)

    if problems_df.empty:
        logger.warning("No problems loaded. Exiting test run.")
        return pd.DataFrame() # Return empty DataFrame

    results_data = [] # List to store results for later logging

    logger.info("\nProcessing problems...")
    for index, row in problems_df.iterrows():
        image_name = row['image_name']
        input_feature = row['input_feature']
        context_feature = row['context_feature']
        correct_answer = str(row['correct_answer']) # Ensure correct answer is treated as string
        full_image_path = os.path.abspath(os.path.join(IMAGE_DIR, image_name))

        logger.info(f"--- Processing {index+1}/{len(problems_df)}: {image_name} (Input: {input_feature}, Context: {context_feature}) ---")

        if not os.path.exists(full_image_path):
            logger.warning(f"  Image file not found: {full_image_path}. Skipping for all models.")
            # --- MODIFICATION: Add entries for each model if image not found ---
            for model_name in AI_MODELS:
                results_data.append({
                    'model_name': model_name, # Added model_name
                    'image_name': image_name,
                    'input_feature': input_feature,
                    'context_feature': context_feature,
                    'correct_answer': correct_answer,
                    'llm_answer': 'Error: Image file not found',
                    'evaluation': 'Skipped'
                })
            continue

        logger.info(f"  Encoding image: {full_image_path}...")
        base64_image = encode_image_to_base64(full_image_path)
        if not base64_image:
            logger.error("  Failed to encode image. Skipping for all models.")
            # --- MODIFICATION: Add entries for each model if encoding failed ---
            for model_name in AI_MODELS:
                results_data.append({
                    'model_name': model_name, # Added model_name
                    'image_name': image_name,
                    'input_feature': input_feature,
                    'context_feature': context_feature,
                    'correct_answer': correct_answer,
                    'llm_answer': 'Error: Image encoding failed',
                    'evaluation': 'Error'
                })
            continue

        # --- MODIFICATION: Iterate over each AI model ---
        for model_name in AI_MODELS:
            logger.info(f"  Testing with model: {model_name}...")
            logger.info("    Sending to OpenRouter API...")
            llm_answer = get_openrouter_response(base64_image, model=model_name)
            logger.info(f"    Expected Answer: {correct_answer}")
            logger.info(f"    LLM Answer ({model_name}): {llm_answer}")

            evaluation_result = evaluate_answer(llm_answer, correct_answer, simple_mode=False)
            logger.info(f"    Evaluation ({model_name}): {evaluation_result}")

            results_data.append({
                'model_name': model_name, # Added model_name
                'image_name': image_name,
                'input_feature': input_feature,
                'context_feature': context_feature,
                'correct_answer': correct_answer,
                'llm_answer': llm_answer,
                'evaluation': evaluation_result
            })
    # --- End MODIFICATION for model iteration ---

    logger.info("\n--- Testing finished. ---")

    results_df = pd.DataFrame(results_data) # --- Create DataFrame here ---

    # --- REMOVED: Save results to CSV for individual run ---
    # if not results_df.empty:
    #     if save_results_to_csv(results_df, RESULTS_CSV_PATH): # Pass DF directly
    #          logger.info("Results DataFrame successfully saved.")
    #     else:
    #          logger.error("Failed to save results DataFrame.")
    # else:
    #     logger.info("No results to save.")

    return results_df # --- Return the DataFrame ---


# --- NEW FUNCTION: Run tests in parallel and aggregate results ---
def run_parallel_tests_and_aggregate(num_runs: int):
    """
    Runs the test function multiple times in parallel and aggregates the results.
    """
    logger.info(f"Starting {num_runs} parallel test runs...")
    all_results_dfs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_runs) as executor:
        futures = [executor.submit(run_test) for _ in range(num_runs)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                single_run_df = future.result()
                if not single_run_df.empty:
                    single_run_df['run_id'] = i # Add a run_id to distinguish runs
                    all_results_dfs.append(single_run_df)
                    logger.info(f"Completed run {i+1}/{num_runs}.")
                else:
                    logger.warning(f"Run {i+1}/{num_runs} produced no results.")
            except Exception as e:
                logger.error(f"Error in parallel run {i+1}/{num_runs}: {e}")

    if not all_results_dfs:
        logger.warning("No results collected from any parallel runs.")
        return pd.DataFrame()

    aggregated_df = pd.concat(all_results_dfs, ignore_index=True)
    logger.info(f"Aggregated results from {len(all_results_dfs)} successful runs. Total rows: {len(aggregated_df)}")

    if not aggregated_df.empty:
        if save_results_to_csv(aggregated_df, AGGREGATED_RESULTS_CSV_PATH):
            logger.info(f"Aggregated results successfully saved to {AGGREGATED_RESULTS_CSV_PATH}")
        else:
            logger.error(f"Failed to save aggregated results DataFrame to {AGGREGATED_RESULTS_CSV_PATH}")
    else:
        logger.info("Aggregated DataFrame is empty. Nothing to save.")

    return aggregated_df

# --- NEW FUNCTION: Analyze aggregated results ---
def analyze_aggregated_results(aggregated_df: pd.DataFrame):
    """
    Performs basic analysis on the aggregated results and logs summaries.
    """
    if aggregated_df.empty:
        logger.info("Aggregated DataFrame is empty. No analysis to perform.")
        return

    logger.info("\n--- Starting Analysis of Aggregated Results ---")

    # Overall evaluation statistics
    overall_evaluation_counts = aggregated_df['evaluation'].value_counts()
    logger.info("\nOverall Evaluation Counts (Aggregated across all models):")
    for eval_type, count in overall_evaluation_counts.items():
        logger.info(f"  {eval_type}: {count}")

    # --- MODIFICATION: Add model-specific overall evaluation ---
    logger.info("Overall Evaluation Counts (Per Model):")
    for model_name, model_group in aggregated_df.groupby('model_name'):
        logger.info(f"  Model: {model_name}")
        model_eval_counts = model_group['evaluation'].value_counts()
        for eval_type, count in model_eval_counts.items():
            logger.info(f"    {eval_type}: {count} ({count / len(model_group) * 100:.2f}%)")

    # Per-problem analysis
    logger.info("\nPer-Problem Analysis (Broken down by Model):")
    # Group by problem first, then by model within each problem
    for image_name, problem_group in aggregated_df.groupby('image_name'):
        logger.info(f"\n  Problem: {image_name}")
        logger.info(f"    Correct Answer: {problem_group['correct_answer'].iloc[0]}") # Assuming correct_answer is the same

        for model_name, model_problem_group in problem_group.groupby('model_name'):
            logger.info(f"    Model: {model_name}")

            # LLM Answer distribution for this model and problem
            llm_answer_counts = model_problem_group['llm_answer'].value_counts()
            logger.info("      LLM Answer Distribution:")
            for answer, count in llm_answer_counts.items():
                # Count occurrences relative to the number of runs for this specific model and problem
                num_runs_for_model_problem = len(model_problem_group)
                logger.info(f'        - "{answer}": {count} occurrence(s) (out of {num_runs_for_model_problem} runs for this model)')

            # Evaluation distribution for this model and problem
            evaluation_counts = model_problem_group['evaluation'].value_counts()
            logger.info("      Evaluation Outcome Distribution:")
            for eval_type, count in evaluation_counts.items():
                num_runs_for_model_problem = len(model_problem_group)
                logger.info(f"        - {eval_type}: {count} occurrence(s) (out of {num_runs_for_model_problem} runs for this model, {count / num_runs_for_model_problem * 100:.2f}%)")
    # --- End MODIFICATION for per-problem, per-model analysis ---

    logger.info("\n--- Aggregated Analysis Finished ---")


if __name__ == "__main__":
    # --- Updated main execution flow ---
    logger.info("Main script execution started.")
    
    # Ensure AI_MODELS is used if defined, otherwise fall back to AI_MODEL for single model runs
    # This part doesn't need changing as run_test now internally uses AI_MODELS
    aggregated_results = run_parallel_tests_and_aggregate(NUM_PARALLEL_RUNS)
    
    if aggregated_results is not None and not aggregated_results.empty:
        analyze_aggregated_results(aggregated_results)
    else:
        logger.warning("No aggregated results were generated. Skipping analysis.")
        
    logger.info("Main script execution finished.") 