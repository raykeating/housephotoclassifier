import requests
import os

base_url = "https://cdn.realtor.ca/listing/TS638147351255430000/reb86/highres/1/x5974031_"
save_dir = os.path.expanduser("./images/")
os.makedirs(save_dir, exist_ok=True)
successful_request = True
i = 2

while successful_request:
    url = base_url + str(i) + ".jpg"
    response = requests.get(url)
    if response.status_code != 200:
        successful_request = False
        break
    with open(os.path.join(save_dir, f"{i}.jpg"), "wb") as f:
        f.write(response.content)
    i += 1
    
    