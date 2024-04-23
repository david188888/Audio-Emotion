import requests
import json


url = '10.191.31.112:8010/recognize/'

def get_data():
    with open('data/111.wav', 'rb') as f:
        audio = {'file': f}
        response = requests.post(url,files=audio)
        
        if response.status_code == 200:
            result = json.loads(response.text)
            print(result)
        else:
            print("Error")
            
            
if __name__ == '__main__':
    get_data()