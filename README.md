# Audio-Emotion
Using ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition to realize speech emotion judgment.


## 1. Introduction
This project is based on the [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition]( 
    https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)

## 2. Installation
```bash
pip install -r requirements.txt
```

## 3. Usage
speech_test.py file is used to test the model with a single audio file.The output is the max score of the emotion.
```bash
python speech_test.py
```

Also, you should make sure your datafolder contains the audio file which is ended with .wav.
you could run the following command to transform the audio file to the .wav file.
```bash
python audio_transfer.py
```

## 4.Emotion List
- neutral
- calm
- happy
- sad
- angry
- disgust
- surprised
- fearful