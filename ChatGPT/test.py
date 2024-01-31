from openai import OpenAI
import time

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "Please answer with 'positive' or 'negative' only!"},
            {"role": "user", "content": "Decide if the following movie review is positive or negative: \nI don't know where to start; the acting, the special effects and the writing are all about as bad as you can possibly imagine. I can't believe that the production staff reached a point where they said, \"Our job is done, time for it's release\". I'm just glad the first two in the series never made it as far as the UK. I would actually recommend watching this film just so you can appreciate how well made most films are.<br /><br />I don't know how any of the other IMDb users could find it scary when the \"terrifying\" dinosaurs waddle down corridors with rubber arms flailing around.\n If the movie review is positive please answer 'positive', if the movie review is negative please answer 'negative'. Make your decision based on the whole text."}        
        ],
        stream=False,
    )
    print(response.choices[0].message.content)
except:
    print("Inference failed, waiting 30 seconds...")
    time.sleep(30)
    