
import pandas as pd
import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch.optim.lr_scheduler as lr_scheduler
import math
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sklearn
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
import argparse
import pickle
from torch.utils.tensorboard import SummaryWriter

vocab_dict = {}
device = ''
EMBEDDING_DIM = 512
NUM_ATTN_HEADS = 8
FF_HIDDEN_DIM = 1024
EPOCHS = 50
BATCH_SIZE = 32
NUM_ENC_LAYERS = 6
LEARNING_RATE = 1e-4
ALPHA = 0.5

def preprocessing(data, test_data=False):
    
    def tokenize(s):
        return s.split()
    
    data['utterances'] = data['utterances'].apply(tokenize)
    
    if not test_data:
        data['IOB Slot tags'] = data['IOB Slot tags'].apply(tokenize)
        data['Core Relations'] = data['Core Relations'].fillna("").apply(lambda x: x.replace('_', '-')).apply(tokenize)
        
    return data

class TestDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.vocab = vocab_dict['vocab']
        self.tag2idx = vocab_dict['tag2idx']
        self.rel2idx = vocab_dict['rel2idx']
        self.unkidx = self.vocab['<unk>']
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = [self.vocab.get(token, self.unkidx) for token in row['utterances']]
        return torch.Tensor(text).long()
    
    def __len__(self):
        return len(self.df)


class JointDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.vocab = vocab_dict['vocab']
        self.tag2idx = vocab_dict['tag2idx']
        self.rel2idx = vocab_dict['rel2idx']
        self.unkidx = self.vocab['<unk>']
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = [self.vocab.get(token, self.unkidx) for token in row['utterances']]
        tags = torch.Tensor([self.tag2idx[tag] for tag in row['IOB Slot tags']]).long()
        rels = row['Core Relations']
        return torch.Tensor(text).long(), tags, torch.Tensor(rels).float()
    
    def __len__(self):
        return len(self.df)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    
    def __init__(self, vocab_size, num_tags, num_rels, d_model, num_attn_heads, ff_hid_dim, num_encoder_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = TransformerEncoderLayer(d_model, num_attn_heads, ff_hid_dim, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.d_model = d_model
        self.vocab = vocab_dict['vocab']
        self.linear = nn.Linear(d_model, 128)
        self.tag_linear = nn.Linear(128, num_tags)
        self.rel_linear = nn.Linear(128, num_rels)
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, src):
        # B - Batch size
        # S - Sequence Length
        # D - Embedding dimension (d_model)
                
        padding_mask = (src == self.vocab['<pad>'])[:, :,]
        pad_mask_rel = (src != self.vocab['<pad>'])
        denom = torch.sum(pad_mask_rel, dim=1, keepdim=True)
        src = self.embedding(src) * math.sqrt(self.d_model) # out shape: B X S X D
        src = self.pos_encoder(src)
        enc_output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        src = self.linear(enc_output)
        # tag_logits = self.softmax(self.tag_linear(src))
        tag_logits = self.tag_linear(src) 


        rel_avg = torch.sum(src * pad_mask_rel.long().unsqueeze(-1), dim = 1, keepdim=False)
        rel_avg = rel_avg / denom
        rel_logits = self.sigmoid(self.rel_linear(rel_avg))
                
        return tag_logits, rel_logits


def collate(batch):
    texts, tags, rels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab_dict['vocab']['<pad>'])
    tags = pad_sequence(tags, batch_first=True, padding_value=-1)
    rels = torch.stack(rels, dim=0)
    
    return texts.to(device), tags.to(device), rels.to(device)

def train_batch(model, texts, tags, rels, optimizer):

    model.train()
    tagging_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    relation_loss_fn = nn.BCELoss()
     
    tag_logits, rel_logits = model(texts)
    
    loss1 = tagging_loss_fn(tag_logits.view(-1, tag_logits.shape[2]), tags.view(-1))
    loss2 = relation_loss_fn(rel_logits, rels)
    
    loss = ALPHA * loss1 + (1 - ALPHA) * loss2
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return loss, loss1, loss2, tag_logits, rel_logits

def compute_metrics(tag_true, tag_pred, rel_true, rel_pred):
    
    unpadded_tag_true = []
    unpadded_tag_pred = []
    
    tag_pred = torch.argmax(tag_pred, dim=2)
    tag_mask = tag_true != -1
    tag_pred = tag_pred.where(tag_mask, -1).tolist()
    tag_true = tag_true.tolist()
    
    for i, _ in enumerate(tag_true):
        unpadded_tag_pred.append([vocab_dict['idx2tag'][tag].replace('_', '-') for tag in tag_pred[i] if tag != -1])
        unpadded_tag_true.append([vocab_dict['idx2tag'][tag].replace('_', '-') for tag in tag_true[i] if tag != -1])
    
    tag_f1 = f1_score(unpadded_tag_true, unpadded_tag_pred, average='micro', scheme=IOB2)
    
    rel_pred = torch.where(rel_pred > 0.5, 1, 0).tolist()
    rel_true = rel_true.long().tolist()
    
    rel_f1 = sklearn.metrics.f1_score(rel_true, rel_pred, average='micro')

    return tag_f1, rel_f1

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            texts, tags, rels = batch
            tag_logits, rel_logits = model(texts)
            tagging_loss = nn.CrossEntropyLoss(ignore_index=-1)
            relation_loss = nn.BCELoss()

            # Compute loss
            tags_flat = tags.view(-1)
            tag_logits_flat = tag_logits.view(-1, tag_logits.size(-1))
            tag_loss = tagging_loss(tag_logits_flat, tags_flat)

            rel_loss = relation_loss(rel_logits, rels)
            loss =  ALPHA * tag_loss + (1 - ALPHA) * rel_loss
            total_loss += loss.item()

    tag_f1, rel_f1 = compute_metrics(tags, tag_logits, rels, rel_logits)
    
    return tag_f1, rel_f1, total_loss / len(dataloader)

def train(model, train_dataloader, dev_dataloader, epochs, model_save_path):
    writer = SummaryWriter()  # Create a SummaryWriter object to write TensorBoard logs
    
    best_dev_f1 = 0.0
    best_epoch = -1

    print('-' * 90)
    print('STARTING TRAINING: ')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.98), eps=1e-9)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=EPOCHS)
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_tag_loss = 0.0
        total_rel_loss = 0.0      
        
        # Training phase
        for idx, batch in enumerate(train_dataloader):
            texts, tags, rels = batch
            batch_loss, tag_loss, rel_loss, tags_pred, rels_pred = train_batch(model, texts, tags, rels, optimizer)
            tag_f1, rel_f1 = compute_metrics(tags, tags_pred, rels, rels_pred)
            
            total_loss += batch_loss.item()
            total_tag_loss += tag_loss.item()
            total_rel_loss += rel_loss.item()
        
        scheduler.step()
        
        # Write training loss and F1 metrics to TensorBoard
        writer.add_scalar('Train Total Loss', total_loss / (idx + 1), epoch)
        writer.add_scalar('Train Tag F1', tag_f1, epoch)
        writer.add_scalar('Train Rel F1', rel_f1, epoch)
        
        print(f'EPOCH {epoch+1}/{epochs}:\nTrain Loss = {total_loss/(idx+1):.5f}, Train Tag F1 = {tag_f1:.5f}, Train Relation F1 = {rel_f1:.5f}')
        
        # Evaluation phase on dev set
        dev_tag_f1_epoch, dev_rel_f1_epoch, dev_loss_epoch = evaluate(model, dev_dataloader)
        
        # Write dev F1 metrics to TensorBoard
        writer.add_scalar('Dev Tag F1', dev_tag_f1_epoch, epoch)
        writer.add_scalar('Dev Rel F1', dev_rel_f1_epoch, epoch)
        writer.add_scalar('Dev Loss', dev_loss_epoch, epoch)
        
        print(f'Dev Loss = {dev_loss_epoch:.5f}, Dev Tag F1 = {dev_tag_f1_epoch:.5f}, Dev Relation F1 = {dev_rel_f1_epoch:.5f}')#, ')
        
        # Check if dev performance improves
        if dev_tag_f1_epoch + dev_rel_f1_epoch > best_dev_f1:
            best_dev_f1 = (dev_tag_f1_epoch + dev_rel_f1_epoch)/2
            best_epoch = epoch
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
    
    print(f'Best model saved at epoch {best_epoch} with combined Dev F1 (tag f1 + rel f1)/2 = {best_dev_f1:.5f}')
    
    writer.close()  # Close the SummaryWriter
   
def populate_map(train_df):
    
    global vocab_dict
    
    vocab = {}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1

    for idx, token in enumerate(train_df['utterances'].explode().unique(), start=2):
        vocab[token] = idx

    vocab_dict['vocab'] = vocab
    
    vocab_dict['idx2token'] = {v:k for k, v in vocab.items()}

    vocab_dict['idx2tag'] = dict(enumerate(train_df['IOB Slot tags'].explode().unique()))
    vocab_dict['tag2idx'] = {v:k for k, v in vocab_dict['idx2tag'].items()}

    vocab_dict['idx2rel'] = dict(enumerate(train_df['Core Relations'].explode().dropna().unique()))
    vocab_dict['rel2idx'] = {v:k for k, v in vocab_dict['idx2rel'].items()}
    
    with open('vocab_dict.pkl', 'wb') as f:
        pickle.dump(vocab_dict, f)


def train_model(data, save_model):
    
    # Read train data
    df = pd.read_csv(data)
    
    # Preprocess train data
    df = preprocessing(df)
    
    # Split train data to train and dev
    train_df, dev_df = train_test_split(df, test_size=0.1, random_state=22)

    # Create dictionaries for vocabulary, tags and relations
    populate_map(train_df)
    
    # Create multilabelbinarizier for core relations
    mlb = MultiLabelBinarizer(classes=list(vocab_dict['rel2idx'].keys()))
    mlb.fit(train_df['Core Relations'])
    train_df['Core Relations'] = mlb.transform(train_df['Core Relations']).tolist()
    dev_df['Core Relations'] = mlb.transform(dev_df['Core Relations']).tolist()
    
    VOCAB_SIZE = len(vocab_dict['vocab'])
    NUM_TAGS = len(vocab_dict['tag2idx'])
    NUM_RELS = len(vocab_dict['rel2idx'])

    train_ds = JointDataset(train_df)
    dev_ds = JointDataset(dev_df)
    
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True)
    dev_dataloader = DataLoader(dev_ds, batch_size=len(dev_df), collate_fn=collate, shuffle=False)

    model = TransformerModel(VOCAB_SIZE, NUM_TAGS, NUM_RELS, EMBEDDING_DIM, NUM_ATTN_HEADS, FF_HIDDEN_DIM, NUM_ENC_LAYERS).to(device)
    
    train(model, train_dataloader, dev_dataloader, EPOCHS, save_model)


def test_model(data, model_path, output_csv):
    
    # Read test data
    test_df = pd.read_csv(data)
    
    # Load vocabulary and model
    global vocab_dict
    with open('vocab_dict.pkl', 'rb') as f:
        vocab_dict = pickle.load(f)
    
    
    VOCAB_SIZE = len(vocab_dict['vocab'])
    NUM_TAGS = len(vocab_dict['tag2idx'])
    NUM_RELS = len(vocab_dict['rel2idx'])
    
    # Load the model
    model = TransformerModel(VOCAB_SIZE, NUM_TAGS, NUM_RELS, EMBEDDING_DIM, NUM_ATTN_HEADS, FF_HIDDEN_DIM, NUM_ENC_LAYERS)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Tokenize and preprocess test data
    test_df = preprocessing(test_df, test_data=True)
    test_dataset = TestDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)#, collate_fn=collate_test, shuffle=False)
    
    predictions = {'utterances': [], 'IOB Slot tags': [], 'Core Relations': []}
    
    with torch.no_grad():
        for batch in test_dataloader:
            texts = batch[0]
            texts = texts.unsqueeze(0)
            texts = texts.to(device)
            
            # Perform inference
            tag_logits, rel_logits = model(texts)
            _, tag_preds = torch.max(tag_logits, dim=2)
            rel_preds = (rel_logits > 0.5).int()
            
            # Convert predictions to tokens
            tags_pred = [vocab_dict['idx2tag'][idx.item()] for idx in tag_preds[0]]
            rels_pred = [vocab_dict['idx2rel'][i] for i, val in enumerate(rel_preds[0]) if val == 1]
            
            tokens = [vocab_dict['idx2token'][idx.item()] for idx in texts[0] if idx.item() != vocab_dict['vocab']['<pad>']]
            
            # Append predictions to the dictionary
            predictions['utterances'].append(' '.join(tokens))
            predictions['IOB Slot tags'].append(' '.join(tags_pred))
            predictions['Core Relations'].append(' '.join(rels_pred))
    
    # Create DataFrame from the predictions
    output_df = pd.DataFrame(predictions)
    
    # Save predictions to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


def main():
    
    parser = argparse.ArgumentParser(description='Joint Multi-task training for IOB Slot tagging and Core Relations extraction')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')
    parser.add_argument('--test', action='store_true', help='Flag to test the model')
    parser.add_argument('--data', type=str, help='Path to the data file')
    parser.add_argument('--save_model', type=str, help='Path to save the trained model')
    parser.add_argument('--model_path', type=str, help='Path to the trained model')
    parser.add_argument('--output', type=str, help='Path to save the output predictions')

    args = parser.parse_args()

    global device
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if args.train:
        # Perform training
        if args.data and args.save_model:
            train_model(args.data, args.save_model)
        else:
            print("Missing data path or save_model path for training.")
    elif args.test:
        # Perform testing
        if args.data and args.model_path and args.output:
            test_model(args.data, args.model_path, args.output)
        else:
            print("Missing data path, model path, or output path for testing.")
    else:
        print("Please specify either --train or --test flag.")
    
    
if __name__ == "__main__":
    main()