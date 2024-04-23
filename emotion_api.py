from fastapi import FastAPI, File, UploadFile
import io
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel
import librosa
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
save_directory = "./model"
sample_rate = 16000
duration = 6

def id2class(id):
    if id == 0:
        return "angry"
    elif id == 1:
        return "fear"
    elif id == 2:
        return "happy"
    elif id == 3:
        return "neutral"
    elif id == 4:
        return "sad"
    else:
        return "surprise"



class HubertClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)
        self.init_weights()

    def forward(self, x):
        outputs = self.hubert(x)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.classifier(x)
        return x

processor = Wav2Vec2FeatureExtractor.from_pretrained(save_directory)
model = HubertForSpeechClassification.from_pretrained("model")
model.to(device)
model.eval()

@app.post("/recognize/")
async def transcribe_audio(file: UploadFile = File()):
    contents = await file.read()
    waveform, sample_rate = librosa.load(io.BytesIO(contents), sr=16000)
    
    # 将音频数据预处理并且传入模型
    
    speech = processor(waveform, padding="max_length", truncation=True, max_length=duration * sample_rate,
                       return_tensors="pt", sampling_rate=sample_rate).input_values
    speech = speech.to(device)
    print("Type of speech:", type(speech))
    print("Contents of speech:", speech)

    with torch.no_grad():
        logit = model(speech)
    
    print("Type of logit:", type(logit))
    print("Contents of logit:", logit)
    
    score = F.softmax(logit, dim=1).detach().cpu().numpy()[0]
    score = score.tolist()
    score = [round(x, 3) for x in score]
    print("Type of score:", type(score))
    print("Contents of score:", score)

    id = torch.argmax(logit).cpu().numpy().item()
    print("Type of id:", type(id))
    print("Contents of id:", id)
    
    result = {"predict": id2class(id), "score": score[id]}
    
    
    
    # write the result to a file
    original_filename = file.filename
    base_filename = original_filename.split(".")[0]
    new_filename = base_filename + "_result.txt"
    with open(new_filename, "w") as f:
        f.write(str(result))
        
    return result
    
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host = '127.0.0.1', port=8010)   