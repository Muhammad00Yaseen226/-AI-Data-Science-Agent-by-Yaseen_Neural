"""
Example usage of the AI Data Cleaning & EDA Agent
"""

from main_agent import DataCleaningAgent

# Example 1: Simple usage with a CSV file
def example_simple():
    """Simple example - just run the full pipeline."""
    print("Example 1: Simple Usage")
    print("-" * 50)
    
    agent = DataCleaningAgent(output_dir="output_example1")
    
    # Run complete pipeline
    agent.run_full_pipeline("path/to/your/data.csv")
    
    print("\n✓ Example 1 completed!\n")


# Example 2: Step-by-step usage with custom parameters
def example_custom():
    """Example with custom cleaning and preprocessing parameters."""
    print("Example 2: Custom Configuration")
    print("-" * 50)
    
    agent = DataCleaningAgent(output_dir="output_example2")
    
    # Load data
    agent.load_data("path/to/your/data.xlsx")
    
    # Clean with custom parameters
    agent.clean_data(
        strategy='auto',
        numeric_strategy='median',
        categorical_strategy='mode',
        threshold=0.3  # Drop columns with >30% missing values
    )
    
    # Preprocess
    agent.preprocess_data(
        encoding_method='onehot',
        scaling_method='standard',
        max_categories=15
    )
    
    # Perform EDA
    agent.perform_eda()
    
    # Create visualizations
    agent.create_visualizations()
    
    # Export
    agent.export_results(filename_prefix="my_cleaned_data")
    
    print("\n✓ Example 2 completed!\n")


# Example 3: Process a folder with multiple files
def example_folder():
    """Example processing a folder with multiple CSV files."""
    print("Example 3: Processing Folder")
    print("-" * 50)
    
    agent = DataCleaningAgent(output_dir="output_example3")
    
    # The agent will automatically merge all CSV files in the folder
    agent.run_full_pipeline("path/to/data_folder/")
    
    print("\n✓ Example 3 completed!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("AI Data Cleaning & EDA Agent - Examples")
    print("=" * 50)
    print("\nNote: Update the file paths in the examples before running.\n")
    
    # Uncomment the example you want to run:
    # example_simple()
    # example_custom()
    # example_folder()

