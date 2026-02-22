from flask import Flask, render_template, request, send_from_directory
from churn_analysis import run_analysis
import os

app = Flask(__name__)

# Home page
@app.route('/')
def index():
    return render_template("index.html")

# Handle file upload and analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return "No file uploaded"

        file = request.files['file']
        if file.filename == '':
            return "No file selected"
            
        filepath = "uploaded.xlsx"
        file.save(filepath)

        # Run analysis
        graphs, results, cleaned_file, histogram, now = run_analysis(filepath)

        return render_template(
            "results.html",
            graphs=graphs,
            results=results,
            cleaned_file=cleaned_file,
            histogram=histogram,
            now=now
        )
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return f"""
        <div style="background: #1a1a2e; color: white; padding: 50px; text-align: center;">
            <h2 style="color: #ff6b6b;">❌ Error Occurred</h2>
            <p style="color: #b0b0b0; margin: 20px 0;">{error_msg}</p>
            <a href="/" style="color: #2e69ff; text-decoration: none; font-size: 1.2rem;">
                ← Back to Upload
            </a>
        </div>
        """
# Download cleaned data
@app.route('/download')
def download():
    return send_from_directory("static", "cleaned_churn_data.xlsx", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)





