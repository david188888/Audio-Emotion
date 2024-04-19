import os 
from pydub import AudioSegment

data_path = './data'
for file in os.listdir(data_path):
    if file.endswith('.wav'):
        continue
    else:
        AudioSegment.from_file(os.path.join(data_path, file)).export(os.path.join(data_path, file.split('.')[0] + '.wav'), format='wav')
        os.remove(os.path.join(data_path, file))
        print(f'{file} has been converted to .wav format')
        
        