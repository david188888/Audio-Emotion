import torch
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, AutoConfig
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import os



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dic = './model'
path = './data/55114876.wav'
config = AutoConfig.from_pretrained(model_dic)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dic)
sampling_rate = feature_extractor.sampling_rate
print(f"the sampling rate is: {sampling_rate}")
model = AutoModelForAudioClassification.from_pretrained(model_dic).to(device)
print('Model loaded')
# print(f"the details of the model are: {model.config}")


# preprocess the audio file
def speech_file_to_array_fn(path, sampling_rate):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    return outputs


if __name__ == '__main__':
    result_list = predict(path, sampling_rate)
    # put out the max score result
    for result in result_list:
        if result['Score'] == max([result['Score'] for result in result_list]):
            print(result)
            break