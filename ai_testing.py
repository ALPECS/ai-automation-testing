import os
import base64
import csv
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from sympy import sympify, SympifyError # We'll uncomment and use this later for robust comparison
import logging

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
AI_MODEL = "openai/gpt-4o"
PROBLEMS_CSV_PATH = "problems.csv"
RESULTS_CSV_PATH = "results_openrouter.csv"
IMAGE_DIR = "calculus_problems"

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

# --- CHANGE: Renamed function slightly for clarity ---
# Function to call OpenRouter API
def get_openrouter_response(base64_image, model=AI_MODEL, max_tokens=500):
    """Sends the image and prompt to the OpenRouter API and returns the text response."""
    prompt='''Solve the calculus problem shown in the image.
    In the last line of your response, include the final answer only, formatted in
    a python string for sympify, no other text. Surround it in ```YOUR_STRING```'''
    
    
    system_rule = (
        "You are a calculus solver.  Return your working if you like, "
        "but the very last line **must** be just one python‑sympify string, "
        "wrapped exactly like  ```YOUR_STRING```. If you are unable to solve the problem, "
        "the last line should be exactly ```Unable to solve```, do not include any other text in the last line."
    )
    
    prompt = "Solve the calculus problem shown in the image."
        
    if not base64_image:
        return "Error: Invalid image data."
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_rule},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=max_tokens,
            stop=["```<answer>```"],   # ← hard stop
            temperature=0
        )        # print(response.choices[0].message.content) # Optional: print raw response
        full_response = response.choices[0].message.content.strip()

        # --- ADD: Extract and clean the last line ---
        lines = full_response.splitlines()
        if not lines:
            logger.warning("Empty response received from API.")
            return "Error: Empty response from API."

        last_line = lines[-1].strip()

        # Remove the surrounding ```<answer>``` tags if present
        
        print(f"Last Line before cleaning:{last_line}")
        
        
        cleaned_last_line = last_line.replace("```", "").replace("<answer>", "").replace("```", "").strip()

        return cleaned_last_line

    except Exception as e:
        # --- CHANGE: Use logger ---
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

    if llm_answer == "Unable to solve":
        return "Unsolvable"

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
        try:
            try:
                expr_llm = sympify(llm_answer)
                expr_correct = sympify(correct_answer)
                if expr_llm.equals(expr_correct): # .equals() performs simplification
                     return "Correct, Answers are mathematically equivalent!"
                else:
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
        # Convert list of dictionaries to DataFrame
        results_df = pd.DataFrame(results_data)

        # Save to CSV
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

# --- CHANGE: Renamed main to run_test and return results ---
def run_test():
    """Runs the full testing process and returns results."""
    logger.info("Starting Calculus Problem Testing (using OpenRouter)...")
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
        # Construct full path relative to the script location or use absolute paths in CSV
        # Assuming image_name in CSV is just the filename like 'problem1.png'
        # and images are stored in IMAGE_DIR
        full_image_path = os.path.abspath(os.path.join(IMAGE_DIR, image_name))

        logger.info(f"--- Processing {index+1}/{len(problems_df)}: {image_name} (Input: {input_feature}, Context: {context_feature}) ---")

        # 1. Check if image exists
        if not os.path.exists(full_image_path):
            logger.warning(f"  Image file not found: {full_image_path}. Skipping.")
            results_data.append({
                'image_name': image_name,
                'input_feature': input_feature,
                'context_feature': context_feature,
                'correct_answer': correct_answer,
                'llm_answer': 'Error: Image file not found',
                'evaluation': 'Skipped'
            })
            continue

        # 2. Encode image
        logger.info(f"  Encoding image: {full_image_path}...")
        base64_image = encode_image_to_base64(full_image_path)
        if not base64_image:
            logger.error("  Failed to encode image. Skipping.")
            results_data.append({
                'image_name': image_name,
                'input_feature': input_feature,
                'context_feature': context_feature,
                'correct_answer': correct_answer,
                'llm_answer': 'Error: Image encoding failed',
                'evaluation': 'Error'
            })
            continue

        # 3. Call OpenRouter API
        logger.info("  Sending to OpenRouter API...")
        llm_answer = get_openrouter_response(base64_image, model=AI_MODEL) # Example model
        logger.info(f"  Expected Answer: {correct_answer}")
        logger.info(f"  LLM Answer     : {llm_answer}")

        # 4. Evaluate response
        evaluation_result = evaluate_answer(llm_answer, correct_answer, simple_mode=False) # Switch to SymPy evaluation
        logger.info(f"  Evaluation     : {evaluation_result}")

        # 5. Store results
        results_data.append({
            'image_name': image_name,
            'input_feature': input_feature,
            'context_feature': context_feature,
            'correct_answer': correct_answer,
            'llm_answer': llm_answer,
            'evaluation': evaluation_result
        })

    logger.info("\n--- Testing finished. ---")

    results_df = pd.DataFrame(results_data) # --- Create DataFrame here ---

    # Save results to CSV
    if not results_df.empty:
        # Use the updated results path
        if save_results_to_csv(results_df, RESULTS_CSV_PATH): # Pass DF directly
             logger.info("Results DataFrame successfully saved.")
        else:
             logger.error("Failed to save results DataFrame.")
    else:
        logger.info("No results to save.")

    return results_df # --- Return the DataFrame ---


if __name__ == "__main__":
    run_test() # --- Call the renamed function ---
