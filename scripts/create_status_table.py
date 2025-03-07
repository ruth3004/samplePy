import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import datetime
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_db import SampleDB

def create_status_table(db_path, sample_list=None):
    # Load the sample database
    sample_db = SampleDB()
    sample_db.load(db_path)

    # If no sample list is provided, use all samples in the database
    if sample_list is None:
        sample_list = list(sample_db.samples.keys())

    # Create a DataFrame from the sample database, filtering for the specified samples
    data = []
    for sample_id in sample_list:
        if sample_id in sample_db.samples:
            row = {'Sample ID': sample_id}
            for step in processing_steps:
                row[step] = sample_db.samples[sample_id].get(step)
            data.append(row)
        else:
            print(f"Warning: Sample {sample_id} not found in the database.")

    df = pd.DataFrame(data)
    df.set_index('Sample ID', inplace=True)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, len(df) * 0.5 + 4))  # Increased figure height
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Rotate column labels and adjust their position
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(rotation=90, ha='left', va='bottom')
            cell.set_height(0.05)  # Decrease height of header row
            cell._text.set_y(-0.15)
        if col == -1:
            cell.set_width(0.2)  # Adjust width of index column

    # Color the cells based on their values
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header
            cell.set_text_props(weight='bold')
            cell.set_text_props(color='black')
        elif col == -1:  # Row labels
            cell.set_text_props(weight='bold')
            cell.set_text_props(color='black')
        else:
            value = df.iloc[row - 1, col]
            if value == 'True':
                cell.set_facecolor('#C6EFCE')  # Light green for completed steps
                cell.set_text_props(color='green')
            else:
                cell.set_facecolor('#FFC7CE')  # Light red for pending steps
                cell.set_text_props(color='red')

    plt.title('Sample Processing Status', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    # Save the figure
    datetime_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    plt.savefig(f'{datetime_str}_sample_processing_status.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Status table saved as 'sample_processing_status.png'")

# List of processing steps
processing_steps = [
    '00_load_experiment',
    '01_register_lm_trials',
    '02_register_lm_trials_lm_stack',
    '03_segment_lm_trials',
    '10_extract_lm_traces',
    '11_normalize_lm_traces',
    '12_deconvolve_lm_traces',
    '13_analyze_correlation_lm_traces',
    '20_preprocess_lm_stack',
    '21_register_lm_stack_to_ref_stack',
    '22_segment_lm_stack_from_em_warped_stack',
    '23_extract_marker_from_channel',
    '30_segment_em_stack',
    '31_segment_glomeruli',
    '32_find_landmarks_with_BigWarp',
    '33_register_em_stack_lm_stack_from_landmarks',
    '34_register_em_stack_lm_trials'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a status table for specified samples or all samples")
    parser.add_argument("--sample_list", help="Path to a text file containing sample IDs (optional)")
    parser.add_argument("--db_path", default=r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv',
                        help="Path to the sample database CSV file")
    args = parser.parse_args()

    if args.sample_list:
        with open(args.sample_list, 'r') as f:
            sample_list = f.read().splitlines()
    else:
        sample_list = None

    create_status_table(args.db_path, sample_list)