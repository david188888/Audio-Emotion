# Audio-Emotion
Using ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition to realize speech emotion judgment.

## 1. Introduction
This project is based on the [xmj2002/hubert-base-ch-speech-emotion-recognition](https://huggingface.co/xmj2002/hubert-base-ch-speech-emotion-recognition)

## 2. Installation
```bash
pip install -r requirements.txt
```

## 3. Usage
speech_test.py file is used to test the model with a single audio file. The output is the max score of the emotion.
    
```bash
python speech_test.py
```
Also, you should make sure your data folder contains the audio file which is ended with .wav.
You could run the following command to transform the audio file to the .wav file.
```bash
python audio_transfer.py
```
Or you could use the following command to transform the audio file to the .wav file.
```bash
ffmpeg -i input.mp3 -acodec pcm_s16le -ac 1 -ar 16000 output.wav
```

## 4.Emotion List
- neutral
- happy
- anger
- fear
- surprise
- sad

## 5. Classifier Implementation Details

### 5.1. Overview
In this project, I enhance the speech emotion recognition model with a custom classifier. The classifier is designed to fine-tune the predictions based on the characteristics of the audio processed.

### 5.2. Classifier Architecture
```python
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
```
#### Functionality
- The HubertClassificationHead is a neural network module that applies a linear transformation, a non-linear activation function, and dropout to manage overfitting. The final output projection adjusts the dimensionality to match the number of emotion classes.
- This structure enables more precise emotion recognition by focusing on optimized feature extraction and classification based on the deep learning insights provided by the Hubert model.


## 6. API Documentation
- Run the following command to start the API server.
```bash
uvicorn emotion_api:app --reload --host 127.0.0.1 --port 8010
```

- The API server will be running on http://127.0.0.1:8010/recognize/
- The API server accepts POST requests with the following parameters:
    - audio: audio file in .wav format
    - Content-Type: multipart/form-data
- The API server will write the result to the file.