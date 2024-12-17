import torch
import pandas as pd

def combine_backtest_split(backtest_split):
    """
    Combine the input, output and cutoff data into a single dataframe
    
    Args:
        backtest_split: A list of three dataframes, the first is the cutoff, the second is the input and the third is the output

    Returns:
        A dataframe with the input, output and cutoff data
    """
    cutoff = backtest_split[0].iloc[0,1]
    input = backtest_split[1].assign(cutoff=cutoff).assign(data_type='input')
    output = backtest_split[2].assign(cutoff=cutoff).assign(data_type='output')
    return pd.concat([input, output])


def create_seq2seq_dataset(df, input_size=180, output_size=14):
    """
    Convert dataframe into sequence-to-sequence PyTorch dataset
    
    Args:
        df: DataFrame with columns ['unique_id', 'ds', 'y', 'cutoff', 'data_type']
        input_size: Length of input sequence
        output_size: Length of target sequence
        
    Returns:
        inputs: Tensor of shape (n_sequences, 1, input_size) 
        targets: Tensor of shape (n_sequences, 1, output_size)
    """
    # Group by cutoff date
    sequences = []
    targets = []
    
    for unique_id in df['unique_id'].unique():
        series_data = df[df['unique_id'] == unique_id]
        for cutoff in series_data['cutoff'].unique():
            # Get data for this window
            window_data = df[df['cutoff'] == cutoff].sort_values('ds')
            
            # Split into input and target sequences
            input_data = window_data[window_data['data_type'] == 'input']['y'].values[-input_size:]
            target_data = window_data[window_data['data_type'] == 'output']['y'].values[:output_size]
            
            # Only keep sequences with full length
            if len(input_data) == input_size and len(target_data) == output_size:
                sequences.append(input_data)
                targets.append(target_data)
        
    # Convert to tensors
    inputs = torch.FloatTensor(sequences).unsqueeze(1)  # Add channel dimension
    targets = torch.FloatTensor(targets).unsqueeze(1)   # Add channel dimension
    
    return inputs, targets