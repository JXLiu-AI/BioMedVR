import json
import os

import requests

# Your Gemini-2.5-Flash API key
API_KEY = "AIzaSyDA2htYvrMmuaZsdpTOTmii2Hh0jMypgbI"
API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key="
    + API_KEY
)

# BUSI classes
busi_classes = ["normal", "benign", "malignant"]

headers = {"Content-Type": "application/json"}


def ask_gemini(prompt):
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        print("Error:", response.text)
        return ""


des_dict = {}
dist_dict = {}
for cname in busi_classes:
    des_prompt = f"Please provide a concise, professional English description for the medical ultrasound image class '{cname}'."
    dist_prompt = f"Please provide a concise, professional English description that highlights the key differences of the medical ultrasound image class '{cname}' compared to other classes in the BUSI dataset."
    des = ask_gemini(des_prompt)
    dist = ask_gemini(dist_prompt)
    des_dict[cname] = des
    dist_dict[cname] = dist
    print(f"{cname} done.")

os.makedirs("attributes/gpt3", exist_ok=True)
with open("attributes/gpt3/busi_des.json", "w") as f:
    json.dump(des_dict, f, indent=2)
with open("attributes/gpt3/busi_dist.json", "w") as f:
    json.dump(dist_dict, f, indent=2)
print("Done!")
