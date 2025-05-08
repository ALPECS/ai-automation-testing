import dash
from dash import html
from dash import dcc
# from dash.dependencies import Input, Output # No longer needed for this version
import plotly.express as px
import pandas as pd
import logging
import dash_bootstrap_components as dbc # Added

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
AGGREGATED_RESULTS_CSV_PATH = "aggregated_results_pydantic.csv"

# --- Load Data ---
def load_data(csv_path):
    """Loads data from the CSV file."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded data from {csv_path}. Shape: {df.shape}")
        essential_columns = ['image_name', 'llm_answer', 'evaluation', 'run_id', 'correct_answer', 'model_name']
        if not all(col in df.columns for col in essential_columns):
            logger.error(f"CSV missing one or more essential columns: {essential_columns}. Available: {df.columns.tolist()}")
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        logger.error(f"Error: Aggregated results file not found at {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading aggregated data: {e}")
        return pd.DataFrame()

aggregated_df = load_data(AGGREGATED_RESULTS_CSV_PATH)

# --- MODIFICATION: Rename models for display ---
if not aggregated_df.empty:
    model_name_map = {
        "google/gemini-2.0-flash-001": "Deepseek",
        "openai/gpt-4o-mini": "ChatGPT"
        # Add other mappings here if needed
    }
    aggregated_df['model_name'] = aggregated_df['model_name'].replace(model_name_map)
    logger.info(f"Model names replaced. Current unique model names: {aggregated_df['model_name'].unique().tolist()}")
# --- END MODIFICATION ---

# --- Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP]) # Modified for DBC
app.title = "LLM Calculus Test Analysis"

# --- App Layout ---
if not aggregated_df.empty:
    # --- MODIFIED: Overall Statistics at a Glance (Now includes Per-Model breakdown) ---
    at_a_glance_section_children = []

    # Global Stats
    overall_total_problems_in_dataset = aggregated_df['image_name'].nunique()
    overall_num_models_evaluated = aggregated_df['model_name'].nunique()

    global_stats_cards = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Total Unique Problems in Dataset"), dbc.CardBody(f"{overall_total_problems_in_dataset}", className="text-center h4")], color="primary", inverse=True, className="mb-2")),
        dbc.Col(dbc.Card([dbc.CardHeader("Total Models Evaluated"), dbc.CardBody(f"{overall_num_models_evaluated}", className="text-center h4")], color="secondary", inverse=True, className="mb-2")),
    ], className="mb-4")
    at_a_glance_section_children.append(global_stats_cards)

    # Per-Model Stats
    at_a_glance_section_children.append(html.H3("Per-Model Performance At a Glance", className="mt-4 mb-3 text-center")) # Centered Subheader

    model_cards_row_items = []
    for model_name in sorted(aggregated_df['model_name'].unique()):
        model_df = aggregated_df[aggregated_df['model_name'] == model_name]
        problems_attempted_by_model = model_df['image_name'].nunique()
        total_runs_model = len(model_df)
        correct_evaluations_model = model_df['evaluation'].str.contains('Correct', case=True, na=False).sum()
        accuracy_model = (correct_evaluations_model / total_runs_model) * 100 if total_runs_model > 0 else 0

        model_stat_card = dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H5(model_name, className="mb-0 text-center")), # Centered Model Name
                dbc.CardBody([
                    html.P(f"Problems Attempted: {problems_attempted_by_model}", className="card-text"),
                    html.P(f"Total Runs: {total_runs_model}", className="card-text"),
                    html.P(f"Accuracy: {accuracy_model:.2f}%", className="card-text fw-bold"), # Bold Accuracy
                ])
            ], className="mb-3 h-100 shadow-sm"), # Added shadow for better card distinction
            lg=4, md=6, sm=12, className="mb-4" # Use mb-4 for consistent spacing with global_stats_cards
        )
        model_cards_row_items.append(model_stat_card)

    at_a_glance_section_children.append(dbc.Row(model_cards_row_items, className="justify-content-center")) # Center the row of model cards if they don't fill it


    # --- Overall Evaluation Summary by Model (Grouped Bar Chart) ---
    overall_eval_summary_df = aggregated_df.groupby(['model_name', 'evaluation']).size().reset_index(name='count')

    # Calculate percentages for overall evaluation summary
    overall_model_totals = overall_eval_summary_df.groupby('model_name')['count'].sum().reset_index(name='total_model_count')
    overall_eval_summary_df = pd.merge(overall_eval_summary_df, overall_model_totals, on='model_name', how='left')
    overall_eval_summary_df['percentage'] = (overall_eval_summary_df['count'] / overall_eval_summary_df['total_model_count']).fillna(0)

    fig_overall_eval = px.bar(overall_eval_summary_df,
                                x='model_name',
                                y='count',
                                color='evaluation',
                                barmode='group',
                                title='Overall Evaluation Outcome Distribution by Model (All Problems & Runs)',
                                labels={'count': 'Number of Outcomes', 'model_name': 'AI Model', 'evaluation': 'Evaluation Outcome'},
                                color_discrete_map={ # General color mapping, can be expanded
                                    outcome: 'green' if 'Correct' in outcome else ('red' if 'Incorrect' in outcome else 'orange')
                                    for outcome in aggregated_df['evaluation'].unique()
                                },
                                custom_data=['percentage']) # Add percentage to custom_data
    fig_overall_eval.update_layout(xaxis_tickangle=-45)
    fig_overall_eval.update_traces(texttemplate='%{y} (%{customdata[0]:.0%})', textposition='outside')


    # --- Create Per-Problem Sections Dynamically ---
    per_problem_layout = [] # Changed to a simple list
    unique_problems = sorted(aggregated_df['image_name'].unique())

    for problem_name in unique_problems:
        problem_df = aggregated_df[aggregated_df['image_name'] == problem_name]
        correct_answer = problem_df['correct_answer'].iloc[0] if not problem_df.empty else "N/A"

        # --- MODIFICATION: Per-Problem Evaluation Outcomes by Model (Grouped Bar Chart) ---
        problem_eval_summary_df = problem_df.groupby(['model_name', 'evaluation']).size().reset_index(name='count')

        # Calculate percentages for per-problem evaluation outcomes
        if not problem_eval_summary_df.empty:
            problem_model_totals = problem_eval_summary_df.groupby('model_name')['count'].sum().reset_index(name='total_model_count')
            problem_eval_summary_df = pd.merge(problem_eval_summary_df, problem_model_totals, on='model_name', how='left')
            problem_eval_summary_df['percentage'] = (problem_eval_summary_df['count'] / problem_eval_summary_df['total_model_count']).fillna(0)
        else:
            # Add empty 'percentage' column if df is empty to avoid issues with custom_data
            problem_eval_summary_df['percentage'] = pd.Series(dtype='float64')


        fig_problem_eval_outcomes = px.bar(problem_eval_summary_df,
                                            x='model_name',
                                            y='count',
                                            color='evaluation',
                                            barmode='group',
                                            title=f'Evaluation Outcomes by Model',
                                            labels={'count': 'Number of Outcomes', 'model_name': 'AI Model'},
                                            color_discrete_map={ # Apply same color logic
                                                outcome: 'green' if 'Correct' in outcome else ('red' if 'Incorrect' in outcome else 'orange')
                                                for outcome in problem_df['evaluation'].unique()
                                            },
                                            custom_data=['percentage']) # Add percentage to custom_data
        fig_problem_eval_outcomes.update_layout(xaxis_tickangle=-45)
        fig_problem_eval_outcomes.update_traces(texttemplate='%{y} (%{customdata[0]:.0%})', textposition='outside')

        # --- REMOVED: Per-Problem LLM Answer Distribution by Model ---
        # llm_answers_sections and related figures/tables are removed.

        problem_section = dbc.Card([
            dbc.CardHeader(html.H4(f"Problem: {problem_name}", className="mb-0")),
            dbc.CardBody([
                html.P(f"Correct Answer: {correct_answer}", style={'fontWeight': 'bold', 'marginBottom': '15px'}),
                dcc.Graph(figure=fig_problem_eval_outcomes)
            ])
        ], className="mb-3") # Added spacing
        per_problem_layout.append(problem_section)
    # --- End Per-Problem Sections ---

    app.layout = dbc.Container(fluid=True, children=[ # Changed to DBC Container
        dbc.Row(dbc.Col(html.H1("LLM Calculus Problem Solving: Aggregated Results Analysis", className="text-center my-4"), width=12)), # Centered title

        *at_a_glance_section_children, # Modified: Unpack the list of components

        dbc.Row(dbc.Col(html.H2("Overall Model Performance Comparison", className="mt-4 mb-3 text-center"), width=12)), # Centered Subheader
        dbc.Card(dbc.CardBody(dcc.Graph(id='overall-evaluation-summary', figure=fig_overall_eval)), className="shadow-sm mb-4"), # Added shadow and margin

        dbc.Row(dbc.Col(html.H2("Per-Problem Analysis", className="mt-5 mb-3 text-center"), width=12)), # Centered Subheader
        *per_problem_layout # Unpack problem sections directly
    ])
else:
    app.layout = dbc.Container(fluid=True, children=[ # Changed to DBC Container
        dbc.Alert(
            [
                html.H1("LLM Calculus Problem Solving: Aggregated Results Analysis", className="alert-heading"),
                html.P(f"Could not load data from {AGGREGATED_RESULTS_CSV_PATH}. Please ensure the file exists and is correctly formatted, then restart the dashboard.")
            ],
            color="danger",
            className="mt-5"
        )
    ])

# --- REMOVED Callbacks for dropdown ---
# @app.callback(
#     [Output('llm-answer-distribution-chart', 'figure'),
#      Output('evaluation-outcome-chart', 'figure'),
#      Output('problem-correct-answer', 'children')],
#     [Input('problem-selector-dropdown', 'value')]
# )
# def update_problem_charts(selected_image_name):
#     if selected_image_name is None or aggregated_df.empty:
#         empty_fig = {'data': [], 'layout': { 'title': 'No data to display'}}
#         return empty_fig, empty_fig, "Correct Answer: Not available"
# 
#     filtered_df = aggregated_df[aggregated_df['image_name'] == selected_image_name]
#     correct_answer = filtered_df['correct_answer'].iloc[0] if not filtered_df.empty else "N/A"
# 
#     # 1. LLM Answer Distribution (Bar Chart)
#     llm_ans_counts = filtered_df['llm_answer'].value_counts().reset_index()
#     llm_ans_counts.columns = ['llm_answer', 'count']
#     fig_llm_answers = px.bar(llm_ans_counts,
#                                x='llm_answer',
#                                y='count',
#                                title=selected_image_name, # Updated title
#                                labels={'llm_answer': 'LLM Answer', 'count': 'Frequency'},
#                                color='llm_answer')
#     fig_llm_answers.update_layout(xaxis_tickangle=-45, showlegend=False)
# 
#     # 2. Evaluation Outcome Distribution (Pie Chart)
#     eval_counts = filtered_df['evaluation'].value_counts().reset_index()
#     eval_counts.columns = ['evaluation', 'count']
#     fig_eval_outcomes = px.pie(eval_counts,
#                                names='evaluation',
#                                values='count',
#                                title=f"Evaluation Outcomes for: {selected_image_name}",
#                                hole=.3)
#     fig_eval_outcomes.update_traces(textinfo='percent+label')
# 
#     correct_answer_display = f"Correct Answer for {selected_image_name}: {correct_answer}"
# 
#     return fig_llm_answers, fig_eval_outcomes, correct_answer_display

# --- Run Server ---
if __name__ == '__main__':
    if not aggregated_df.empty:
        logger.info("Starting Dash server...")
        app.run(debug=False, host='0.0.0.0')
    else:
        logger.error("Failed to load data. Dash server will not start.")
        print(f"Error: Could not load data from {AGGREGATED_RESULTS_CSV_PATH}. Ensure the file exists and contains data.")
        print("Run the ai_testing_pydantic.py script first to generate the aggregated_results_pydantic.csv file.") 