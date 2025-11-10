🧠 AI Data Cleaning & EDA Agent

An intelligent Python agent that automatically cleans, preprocesses, analyzes, visualizes datasets, and optionally runs AutoML—all with minimal user effort. Now comes with a Streamlit web interface for easy interaction and showcasing.

✨ Key Features

📂 Multi-format Support: CSV, Excel, JSON, TXT, ZIP (automatically merges multiple files)

🧹 Automatic Data Cleaning: Handles missing values, duplicates, wrong data types, and outliers

⚙️ Smart Preprocessing: Encodes categorical variables, scales numeric features, normalizes data

📊 Comprehensive EDA: Generates statistical summaries, correlation analysis, and distribution insights

📈 Rich Visualizations: Histograms, heatmaps, boxplots, pairplots, missing value plots, categorical counts, and more

🤖 Optional AutoML: Trains multiple models automatically, evaluates regression (RMSE, MAE, R²) or classification (accuracy, F1-score), and recommends the best model

💻 Interactive Web UI: Streamlit interface for file upload, parameter customization, pipeline execution, and result visualization

💾 Export Ready: Saves cleaned datasets, EDA reports, visualizations, and AutoML metrics

🚀 Quick Start
Installation

# Activate your environment
conda activate dataagent

# Install required dependencies
pip install -r requirements.txt

# Install Streamlit if not already installed
pip install streamlit


🔹 Run via Python Script

from main_agent import DataCleaningAgent

# Initialize the agent
agent = DataCleaningAgent(output_dir="output")

# Run the full pipeline
agent.run_full_pipeline("path/to/your/data.csv", run_automl=True, automl_target="target_column")


🔹 Run via Command Line

# Full pipeline with optional AutoML
python main_agent.py path/to/data.csv --output-dir output --automl --automl-target target_column


🔹 Run with Streamlit Web UI

streamlit run app.py

Upload your dataset

Customize cleaning, preprocessing, EDA, and AutoML parameters

Run the pipeline interactively

View plots, reports, and metrics directly in the browser


📁 Output Structure

output/
├── cleaned_data.csv          # Cleaned and preprocessed dataset
├── eda_report.txt            # Full EDA report
├── automl_results.json       # Optional AutoML results
└── plots/
    ├── histograms.png
    ├── correlation_heatmap.png
    ├── boxplots.png
    ├── pairplot.png
    ├── missing_values.png
    ├── categorical_counts.png
    └── distributions.png

🔧 Module Structure

file_handler.py: Reads various file types and merges multiple files

data_cleaner.py: Cleans missing values, duplicates, types, and outliers

preprocessor.py: Encodes, scales, and normalizes data

eda_analyzer.py: Performs statistical EDA

visualizer.py: Generates rich visualizations

main_agent.py: Core orchestrator

app.py: Streamlit-based interactive web interface

⚙️ Customizable Options
Data Cleaning

strategy: 'auto', 'drop', 'fill', 'threshold'

numeric_strategy: 'mean', 'median', 'mode', 'forward_fill'

categorical_strategy: 'mode', 'forward_fill', 'unknown'

threshold: Drop columns/rows with > threshold missing fraction (0–1)

Preprocessing

encoding_method: 'auto', 'onehot', 'label', 'ordinal'

scaling_method: 'standard', 'minmax', 'robust', None

max_categories: Max categories for one-hot encoding (default: 10)

Outlier Handling

method: 'cap', 'remove', 'transform'

outlier_method: 'iqr' or 'zscore'

threshold: Multiplier for IQR/Z-score (default: 1.5)

📊 EDA Report Contents

Dataset info: shape, data types, memory usage

Missing value analysis

Numeric columns summary: mean, median, std, quartiles, skewness, kurtosis

Categorical columns summary: unique counts, top values

Correlation analysis and high correlation detection

Distribution analysis

🎨 Visualizations

Histograms (numeric columns)

Correlation Heatmap

Boxplots (outliers)

Pairplot (numeric relationships)

Missing Values Plot

Categorical Counts

Distribution Plots (KDE with mean & median)

🔍 Supported File Formats

CSV (.csv)

Excel (.xlsx, .xls)

JSON (.json)

TXT (.txt) - Auto-detect delimiter

ZIP (.zip) - Extracts & merges all CSVs

📝 Requirements

Python 3.8+

pandas >= 2.0

numpy >= 1.24

matplotlib >= 3.7

seaborn >= 0.12

scikit-learn >= 1.7.2

openpyxl >= 3.1

streamlit >= 1.5

🤝 Contributing

Contributions are welcome! Submit Pull Requests or open issues for new features or bug fixes.

📄 License

MIT License. Open-source and free to use.

🙏 Acknowledgments

Built with: pandas, numpy, matplotlib, seaborn, scikit-learn, Streamlit

Happy Data Cleaning & AI Exploration! 🎉

Developed by "Muhammad Yaseen"
(Rising AI/ML Engineer /Data Scientist)

