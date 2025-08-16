"""
Generating BERT embeddings from diagnosis text per ICU stay
"""
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import transformers as ppb
from src.utils import load_json


def save_bert_tokens(data_dir, graph_dir, max_len=512):
    """
    Read cleaned diagnosis text and tokenize.
    """
    # Ensure graph directory exists
    os.makedirs(graph_dir, exist_ok=True)
    
    # read diagnosis strings
    dfs = []
    for split in ['train', 'val', 'test']:
        data = pd.read_csv(data_dir + split + '/diagnosis_strings_cleaned.txt', sep="\n", header=None)
        data.columns = ["sentence"]
        dfs.append(data)
    df = pd.concat(dfs, ignore_index=True)
    
    # Download pre-trained weights from BERT. This takes a while if running for the first time. 
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    tokenized = df['sentence'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    # Cut off tokens at 512 and pad. 
    padded = []
    for i in tokenized.values:
        if len(i) < max_len:
            p = np.array(i + [0]*(max_len-len(i)))
        else:
            p = np.array(i[:max_len])
        padded.append(p)
    padded = np.array(padded)
    attention_mask = np.where(padded != 0, 1, 0)

    # Get the actual number of samples
    num_samples = padded.shape[0]
    
    # Saving padded.dat and attention_mask.dat
    print(f'saving data with shape {padded.shape}..')
    save_path = graph_dir + 'padded.dat'
    write_file = np.memmap(save_path, dtype=int, mode='w+', shape=(num_samples, max_len))
    write_file[:, :] = padded
    save_path = graph_dir + 'attention_mask.dat'
    write_file = np.memmap(save_path, dtype=int, mode='w+', shape=(num_samples, max_len))
    write_file[:, :] = attention_mask
    return padded, attention_mask, num_samples


def read_data(data_dir, graph_dir, gpu=True):
    """
    Read tokens and attention masks as input to BERT model.
    """
    # Check for both files
    padded_path = graph_dir + 'padded.dat'
    mask_path = graph_dir + 'attention_mask.dat'
    
    if os.path.exists(padded_path) and os.path.exists(mask_path):
        print("Found existing token files")
        # Get the file size to determine the shape
        padded_size = os.path.getsize(padded_path)
        num_samples = padded_size // (512 * np.dtype(int).itemsize)
        
        padded = np.memmap(padded_path, dtype=int, shape=(num_samples, 512))
        attention_mask = np.memmap(mask_path, dtype=int, shape=(num_samples, 512))
    else:
        print("One or both token files not found. Generating new tokens...")
        padded, attention_mask, num_samples = save_bert_tokens(data_dir, graph_dir)
    
    input_ids = torch.tensor(padded)
    attn_mask = torch.tensor(attention_mask)

    if gpu:
        input_ids = input_ids.to('cuda')
        attn_mask = attn_mask.to('cuda')
    return input_ids, attn_mask, num_samples


def run_bert_in_mini_batches(graph_dir, model, input_ids, attn_mask, num_samples, bsz, gpu=True):
    """
    Pass the prepared tokens in mini-batches to a pre-trained BERT model and save its output.
    """
    save_path = graph_dir + 'bert_out.npy'
    write_file = np.zeros((num_samples, 768))
    model.eval()
    if gpu:
        model.to('cuda')
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, bsz)):
            batch_input = input_ids[i: i+bsz]
            batch_mask = attn_mask[i: i+bsz]
            actual_bsz = len(batch_input)
            last_hidden_states = model(batch_input, attention_mask=batch_mask)
            out = last_hidden_states[0][:,0,:].detach().cpu().numpy()
            write_file[i: i+actual_bsz] = out
            if i % 1000 == 0:
                print(f"Processed {i}/{num_samples} samples")
    with open(save_path, 'wb') as f:
        np.save(f, write_file)


def main(data_dir, graph_dir, gpu=True):
    """
    Generate BERT embeddings from cleaned diagnosis text.
    
    :data_dir: where the diagnosis string files are located
    :graph_dir: where to output the bert embeddings
    """
    print(f"Using data_dir: {data_dir}")
    print(f"Using graph_dir: {graph_dir}")
    
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    model = model_class.from_pretrained(pretrained_weights)
    input_ids, attn_mask, num_samples = read_data(data_dir, graph_dir, gpu)
    print(f"Processing {num_samples} samples")
    run_bert_in_mini_batches(graph_dir, model, input_ids, attn_mask, num_samples, bsz=20, gpu=gpu)


if __name__ == '__main__':
    try:
        paths = load_json('paths.json')
        data_dir = paths['MIMIC_path']
        graph_dir = paths['graph_dir']
        
        # Print paths for debugging
        print(f"Data directory: {data_dir}")
        print(f"Graph directory: {graph_dir}")
        
        # Ensure paths end with trailing slash
        if not data_dir.endswith('/'):
            data_dir += '/'
        if not graph_dir.endswith('/'):
            graph_dir += '/'
            
        # Ensure graph directory exists
        os.makedirs(graph_dir, exist_ok=True)
        print(f"Ensured graph directory exists: {graph_dir}")
        
        # Check for existing files and remove if only one exists
        padded_path = graph_dir + 'padded.dat'
        mask_path = graph_dir + 'attention_mask.dat'
        
        if os.path.exists(padded_path) and not os.path.exists(mask_path):
            print("Found padded.dat but not attention_mask.dat. Removing padded.dat to regenerate both.")
            os.remove(padded_path)
        elif os.path.exists(mask_path) and not os.path.exists(padded_path):
            print("Found attention_mask.dat but not padded.dat. Removing attention_mask.dat to regenerate both.")
            os.remove(mask_path)
            
        # Run the main function
        main(data_dir, graph_dir)
    except Exception as e:
        import traceback
        print(f"Error encountered: {e}")
        print(traceback.format_exc())