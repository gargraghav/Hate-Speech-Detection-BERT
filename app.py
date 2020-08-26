import numpy as np
import pandas as pd
import torch

import transformers
from transformers import BertForSequenceClassification, BertConfig
from transformers import BertTokenizer

# from keras.preprocessing.sequence import pad_sequences

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

from tqdm import tqdm, trange,tqdm_notebook

#web frameworks
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import uvicorn
import aiohttp
import asyncio
import os
import sys
MAX_SEQUENCE_LENGTH = 42
BERT_MODEL ='bert-base-uncased'
app = Starlette()

def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)


device = 'cpu'
# print('Path is')
# print(Path(''))
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
print('Tokenizer loaded.')



print('Model is loading...')

path = 'bert_pytorch_e_1.bin'

model_state_dict = torch.load(path, map_location=torch.device('cpu'))

print('Local model is initializing...')
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, state_dict=model_state_dict) 
print('Local model is initializing...')

model.to(device)



# async def get_message(text):
#     async with aiohttp.ClientSession() as session:
#         async with session.get(text) as response:
#             return await response.read()

@app.route("/")

def form(request):
    return HTMLResponse(
            """
            <h1> Hate Speech Detection </h1>
            <br>
            <u> Submit Text </u>
            <form action = "/classify-text" method="get">
                1. <input type="text" name="text" size="60"><br><p>
                2. <input type="submit" value="Submit">
            </form>
            """) 


@app.route("/form")
def redirect_to_homepage(request):
        return RedirectResponse("/")

@app.route('/classify-text', methods = ["GET"])
def classify_text(request):
    message = request.query_params["text"]
    return predict(message)

def predict(message):
    # message = request.get_json(force=True)
    # message = request.get_json(force=True)
    sentence = message
    print(sentence)
    lines = [sentence]
    lines_df = pd.DataFrame (lines,columns=['Text'])

    val_sequences = convert_lines(lines_df['Text'],MAX_SEQUENCE_LENGTH,tokenizer)
    X_val = val_sequences

    valid_preds_test = np.zeros((len(X_val)))
    valid_test = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.long))
    valid_loader_test = torch.utils.data.DataLoader(valid_test, batch_size=32, shuffle=False)

    tk0 = tqdm_notebook(valid_loader_test)
    for i,(x_batch,)  in enumerate(tk0):
        pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        pred = pred[0]
        y_soft = F.softmax(pred,dim=1)
        valid_preds_test[i*32:(i+1)*32]=y_soft[:,1].detach().cpu().squeeze().numpy()

    print(valid_preds_test[0])
    toxic_proba = valid_preds_test[0] * 100
    toxic_proba = np.round(toxic_proba, 2) 
    toxic_proba = str(toxic_proba)
    
    print(toxic_proba)
    # response = {'greeting': toxic_proba + '%'}
    
    return HTMLResponse(
        """
        <html>
            <body>
                <p> Toxicity: <b> %s </b> </p>
            </body>
        </html>
        """ %(toxic_proba))



if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008)) 
        uvicorn.run(app, host = "0.0.0.0", port = port)