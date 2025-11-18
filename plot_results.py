# Generates visualisations from a training results CSV file.

import argparse
import logging
import os

import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_bar, labs, theme, ggsave, facet_wrap, element_line, scale_y_continuous, \
    element_blank

RESULTS_DIR = "results"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def format_label(label, max_length=20):
    label = label.replace('.txt', '')
    return label
    # return '\n'.join(textwrap.wrap(label, max_length))

def create_visualisations(results_df):
    if results_df.empty:
        logging.warning("No results to visualise.")
        return

    # Apply label formatting
    results_df['input_file'] = results_df['input_file'].apply(format_label)

    # Overall summary plot
    summary_df_overall = results_df.groupby(['model', 'input_file']).agg(
        mean_time=('time', 'mean'),
        std_time=('time', 'std')
    ).reset_index()

    max_time_overall = summary_df_overall['mean_time'].max()
    
    p_overall = (ggplot(summary_df_overall, aes(x='model', y='mean_time', fill='model'))
         + geom_bar(stat='identity', position='dodge')
         + facet_wrap('~input_file', ncol=3, scales='fixed')
         + labs(title='Overall Average Training Time by Model and Input File',
                x='Model',
                y='Average Training Time (seconds)',
                fill='Model')
         + theme(axis_text_x=element_blank(),
                 panel_grid_major_y=element_line(color='white', size=0.25),
                 legend_position='right')
         + scale_y_continuous(expand=(0, 0), breaks=np.arange(0, max_time_overall + 20, 20)))

    output_filename_overall = os.path.join(RESULTS_DIR, "training_visualisation_overall.png")
    ggsave(p_overall, filename=output_filename_overall, dpi=300, width=12, height=6)
    logging.info(f"Overall visualisation saved to {output_filename_overall}")


    params_to_plot = ['creativity_temperature', 'batch_size', 'training_epochs', 'gradient_accumulation_steps', 'sanitise']

    for param in params_to_plot:
        if param not in results_df.columns:
            logging.warning(f"Parameter '{param}' not found in results, skipping visualisation.")
            continue
            
        # Calculate mean and standard deviation
        summary_df = results_df.groupby(['model', 'input_file', param]).agg(
            mean_time=('time', 'mean'),
            std_time=('time', 'std')
        ).reset_index()
        
        max_time_param = summary_df['mean_time'].max()

        p = (ggplot(summary_df, aes(x=param, y='mean_time', fill='model'))
             + geom_bar(stat='identity', position='dodge')
             + facet_wrap('~input_file', ncol=3, scales='fixed')
             + labs(title=f'Average Training Time by {param.replace("_", " ").title()}',
                    x=None,
                    y='Average Training Time (seconds)',
                    fill='Model')
             + theme(axis_text_x=element_blank(),
                     axis_ticks_x=element_blank(),
                     panel_grid_major_y=element_line(color='gray', size=0.25),
                     legend_position='right')
             + scale_y_continuous(expand=(0, 0), breaks=np.arange(0, max_time_param + 20, 20)))

        output_filename = os.path.join(RESULTS_DIR, f"training_visualisation_by_{param}.png")
        ggsave(p, filename=output_filename, dpi=300, width=18, height=6)
        logging.info(f"Visualisation saved to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate visualisations from training results.")
    parser.add_argument("csv_file", type=str, help="Path to the training results CSV file.")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        logging.error(f"CSV file not found at '{args.csv_file}'")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results_df = pd.read_csv(args.csv_file)
    create_visualisations(results_df)

if __name__ == "__main__":
    main()
