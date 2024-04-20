import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel

model_name_or_path = "xmj2002/hubert-base-ch-speech-emotion-recognition"
duration = 6
sample_rate = 16000

config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
)


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


def predict(path, processor, model):
    speech, sr = librosa.load(path=path, sr=sample_rate)
    speech = processor(speech, padding="max_length", truncation=True, max_length=duration * sr,
                       return_tensors="pt", sampling_rate=sr).input_values
    with torch.no_grad():
        logit = model(speech)
    score = F.softmax(logit, dim=1).detach().cpu().numpy()[0]
    id = torch.argmax(logit).cpu().numpy()
    print(f"file path: {path} \t predict: {id2class(id)} \t score:{score[id]} ")


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


processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
# model = HubertForSpeechClassification.from_pretrained(
#     model_name_or_path,
#     config=config,
# )

# model.save_pretrained("model")
model = HubertForSpeechClassification.from_pretrained("model")
model.eval()

# print(f"detail of model: {model}")

path = 'data/111.wav'
predict(path, processor, model)
