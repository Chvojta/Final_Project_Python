import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from savings import (
    calculate_all_metrics, 
    analyze_data, 
    detect_available_analyses, 
    calculate_all_metrics_with_breakdown
)
import numpy as np
import json
import pandas as pd
import uuid

"""
Initiate the flask app and set the secret key and upload folder.
"""
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")
app.config['UPLOAD_FOLDER'] = 'uploads'

"""
This is a JSON encoder that let's us take in the numpy arrays and convert them to lists, which we need for the JSON response.
"""
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

"""
This function reads the Excel file from the path stored in session and returns the DataFrames with the input data that we use to calculate all of the metrics.
"""
def get_dataframes_from_session():
    if 'input_file_path' not in session or not session.get('file_uploaded', False):
        print("No valid file path in session")
        return {}
    file_path = session['input_file_path']
    print(f"Reading file from: {file_path}")
    df_inputs = pd.read_excel(file_path, sheet_name=None)
    return df_inputs

"""
This is the welcome page that we show when the user first opens the app.
"""
@app.route("/")
def welcome():
    return render_template("welcome.html")

"""
This is the page that we show when the user first opens the app, it checks if the file is uploaded and if not, it shows the upload page.
"""
@app.route("/get-started")
def get_started():
    # If session variable doesn't exist, initialize it
    if 'file_uploaded' not in session:
        session['file_uploaded'] = False
    return render_template("get_started.html")

"""
This page runs after the user uplaods the file and we have the data in the session, it is the template for letting them select the strategy / level of aggressiveness they want in looking for value creation opportunities.
"""
@app.route("/select-strategy")
def select_strategy():
    return render_template("select_strategy.html")

"""
This page runs after the user selects the strategy and we have the data in the session, it is the template for letting them select the analyses they want to run. 
It defaults to all analyses, but they can unselect some if they don't want them. If for whatever reason something goes south, it defaults to medium as a strategy.
"""
@app.route("/select-analyses", methods=["POST"])
def select_analyses():
    # Get strategy from previous page
    strategy = request.form.get("strategy", "Medium")
    
    # Get available analyses based on session data
    df_inputs = get_dataframes_from_session()
    available_analyses = detect_available_analyses(df_inputs=df_inputs)
    
    return render_template("select_analyses.html", 
                          strategy=strategy,
                          available_analyses=available_analyses)

"""
This page is the upload functionality, it clears any previous upload data, checks for a file in the request, and then stores the file in the session. It returns to the get_started page with a success or error message.
"""
@app.route("/upload", methods=["POST"])
def upload():
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    
    # Store file with session ID
    file = request.files["file"]
    session_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"inputs_{session_id}.xlsx")
    
    try:
        # Save the file and store path in sessin
        file.save(session_filepath)
        session['input_file_path'] = session_filepath
        session['file_uploaded'] = True
        
        flash("File uploaded successfully!", "success")
    except Exception as e:
        flash(f"Error saving file: {str(e)}", "error")
    
    return redirect(url_for("get_started"))

"""
This page is the results page, it starts by getting the dataframes from the session. If the user has selected the analyses, it will run the analyses and show the results. It passes them to the HTML file that displays them in a table-like format with some excel-like styling.
"""
@app.route("/results", methods=["POST", "GET"])
def results():
    """Display the results page with original and improved income statements."""
    # Get the dataframes from session
    df_inputs = get_dataframes_from_session()
    
    if request.method == "POST":
        strategy = request.form.get("strategy", "Medium")
        selected_analyses = request.form.getlist("analyses")
        session['strategy'] = strategy
        session['selected_analyses'] = selected_analyses
        
        # Get formatted data with selected analyses
        row_labels, years, original_values, improved_values = analyze_data(
            strategy=strategy, 
            selected_analyses=selected_analyses,
            df_inputs=df_inputs
        )
        
        # Get the actual analyses information
        available_analyses = detect_available_analyses(df_inputs=df_inputs)
        selected_analysis_info = {k: available_analyses[k] for k in selected_analyses if k in available_analyses}
        
        return render_template("results.html", 
                             years=years,
                             row_labels=row_labels,
                             original_values=original_values,
                             improved_values=improved_values,
                             selected_analyses=selected_analysis_info)
    else:
        # For GET requests, use stored session data if available
        strategy = session.get('strategy', 'Medium')
        selected_analyses = session.get('selected_analyses', [])
    
    # Get formatted data with selected analyses
    row_labels, years, original_values, improved_values = analyze_data(
        strategy=strategy, 
        selected_analyses=selected_analyses,
        df_inputs=df_inputs
    )
    
    # Get the actual analyses information
    available_analyses = detect_available_analyses(df_inputs=df_inputs)
    selected_analysis_info = {k: available_analyses[k] for k in selected_analyses if k in available_analyses}
    
    return render_template("results.html", 
                         years=years,
                         row_labels=row_labels,
                         original_values=original_values,
                         improved_values=improved_values,
                         selected_analyses=selected_analysis_info)

@app.route('/export_ebitda_bridge')
def export_ebitda_bridge_route():
    """
    Route to export the analysis results to Excel without saving to disk.
    """
    # Get strategy and selected analyses from session
    strategy = session.get('strategy', 'Medium')
    selected_analyses = session.get('selected_analyses', [])
    
    # Check if there's data to export
    if not selected_analyses:
        return "No analyses selected. Please select analyses on the main page first.", 400
    
    try:
        # Generate the metrics data
        df_inputs = get_dataframes_from_session()
        metrics = calculate_all_metrics_with_breakdown(strategy, selected_analyses, df_inputs)
        
        # Import the export function
        from export_metrics import export_ebitda_bridge
        
        # Export to in-memory Excel file
        excel_data = export_ebitda_bridge(metrics)
        
        # Return the Excel file for download directly from memory
        return send_file(
            excel_data,
            as_attachment=True,
            download_name="Optimization_Results.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        return f"Error exporting results: {str(e)}", 500

@app.route("/download-template")
def download_template():
    """
    Route to download the input template Excel file.
    """
    try:
        # Path to the template file / ensure file exists
        template_path = os.path.join(os.path.dirname(__file__), 'Inputs_Template.xlsx')
        if not os.path.exists(template_path):
            flash("Template file not found.", "error")
            return redirect(url_for("get_started"))
        
        # Return the file for download
        return send_file(
            template_path,
            as_attachment=True,
            download_name="Inputs_Template.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        flash(f"Error downloading template: {str(e)}", "error")
        return redirect(url_for("get_started"))

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)