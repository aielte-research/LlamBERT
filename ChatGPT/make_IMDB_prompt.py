import fire
import os
import json
import sys
sys.path.append("..")
from utils import my_open
from typing import Union

def main(
   input_fpath: str="/home/projects/DeepNeurOntology/IMDB_data/promt_eng.json",
   output_folder: str="model_inputs/IMDB/",
   output_fname: Union[str,None]=None,
   shots: int=0,
   **kwargs
):
   if len(kwargs) > 0:
      raise ValueError(f"Unknown argument(s): {kwargs}")
   
   if output_fname is None:
      output_fname = os.path.basename(input_fpath).split(".")[0]

   with open(input_fpath) as f:
      reviews = json.loads(f.read())
   
   prompts=[]
   labels=[]
   for rev in reviews:
      if isinstance(rev, str):
         rev_str=rev
      else:
         rev_str=rev["txt"]

      if shots==0:
         prompts.append([
            {"role": "system", "content": "Please answer with 'positive' or 'negative' only!"},
            {"role": "user", "content": f"Decide if the following movie review is positive or negative: \n{rev_str}\n If the movie review is positive please answer 'positive', if the movie review is negative please answer 'negative'. Make your decision based on the whole text."}
         ])            
      elif shots==1:
         prompts.append([
            {"role": "system", "content": "Please answer with 'positive' or 'negative' only!"},
            {"role": "user", "content": "Decide if the following movie review is positive or negative: \nI don't know where to start; the acting, the special effects and the writing are all about as bad as you can possibly imagine. I can't believe that the production staff reached a point where they said, \"Our job is done, time for it's release\". I'm just glad the first two in the series never made it as far as the UK. I would actually recommend watching this film just so you can appreciate how well made most films are.<br /><br />I don't know how any of the other IMDb users could find it scary when the \"terrifying\" dinosaurs waddle down corridors with rubber arms flailing around.\n If the movie review is positive please answer 'positive', if the movie review is negative please answer 'negative'. Make your decision based on the whole text."},
            {"role": "assistant", "content": "negative"},
            {"role": "system", "content": "Please answer with 'positive' or 'negative' only!"},
            {"role": "user", "content": f"Decide if the following movie review is positive or negative: \n{rev_str}\n If the movie review is positive please answer 'positive', if the movie review is negative please answer 'negative'. Make your decision based on the whole text."}
         ])
      elif shots==2:
         prompts.append([
            {"role": "system", "content": "Please answer with 'positive' or 'negative' only!"},
            {"role": "user", "content": "Decide if the following movie review is positive or negative: \nI don't know where to start; the acting, the special effects and the writing are all about as bad as you can possibly imagine. I can't believe that the production staff reached a point where they said, \"Our job is done, time for it's release\". I'm just glad the first two in the series never made it as far as the UK. I would actually recommend watching this film just so you can appreciate how well made most films are.<br /><br />I don't know how any of the other IMDb users could find it scary when the \"terrifying\" dinosaurs waddle down corridors with rubber arms flailing around.\n If the movie review is positive please answer 'positive', if the movie review is negative please answer 'negative'. Make your decision based on the whole text."},
            {"role": "assistant", "content": "negative"},
            {"role": "system", "content": "Please answer with 'positive' or 'negative' only!"},
            {"role": "user", "content": "Decide if the following movie review is positive or negative: \nEverybody knows that Gregory Widen's original \"The Prophecy\" didn't really require a sequel, but you also don't need a degree in rocket science hanging above your chimney to realize that further cash-ins on this profitable horror concept were inevitable. Part two is a very prototypic example of a straight-to-video sequel, meaning the creative and convoluted plot of the original has been simplified a lot in favor of more action, more witty one-liners and a lot more eerie religious scenery. The only good news is that the producers managed to keep Christopher Walken for the role of Gabriel, and he delivers another gloriously brazen performance that promptly justifies the price of a rental. If it wasn't for Walken's performance (and perhaps a couple of players in the supportive cast like Brittany Murphy and Glenn Danzig), \"The Prophecy II\" surely would have disappeared into oblivion straight after its release. The movie begins with Gabriel literally getting spat out of hell to proceed with his ongoing War of Heaven here on earth. The purpose of his battle this time is to prevent the baby of nurse Valerie Rosales (Jennifer Beals) from getting born. For you see, her unborn child is the first ever hybrid between a heavenly angel and an earthly \"monkey\" and the birth of such a superior being would imply the downfall of Gabriel's evil dominion. Thus, just as in the first movie, he engages a suicidal accomplice to assist him and hunts Valerie all the way down to the Eden for the final showdown. \"The Prophecy II\" is an endurable and occasionally even entertaining movie as long as you don't make comparisons with the original and as long as you manage to overlook the multiple plot holes and errors in continuity. Whenever the storyline becomes too tedious, the makers luckily enough always insert a near-brilliant Christopher Walken moment to distract you. His interactions with the rebellious Izzy and particularly his ignorance regarding modern earthly technologies often result in worthwhile and memorable sequences. On a slightly off-topic note, I often felt like \"The Prophecy II\" ambitions to look similar to \"Terminator II\" \u0085 Gabriel's resurrection looked somewhat like the teleportation of a futuristic cyborg and the Eden location, where the final battle takes place, looks very similar to the steel factory where \"Terminator II\" ended as well. Coincidence, I guess? Overall, this is an inferior and passable sequel but still worth checking out in case you're a fan of Christopher Walken's unique acting charisma (and who isn't?)\n If the movie review is positive please answer 'positive', if the movie review is negative please answer 'negative'. Make your decision based on the whole text."},
            {"role": "assistant", "content": "negative"},
            {"role": "system", "content": "Please answer with 'positive' or 'negative' only!"},
            {"role": "user", "content": f"Decide if the following movie review is positive or negative: \n{rev_str}\n If the movie review is positive please answer 'positive', if the movie review is negative please answer 'negative'. Make your decision based on the whole text."}
         ])
      else:
         print("Only options 0, 1 and 2 shot are implemented.")

      if isinstance(rev, dict) and "label" in rev:
         labels.append(rev["label"])
   
   with my_open(os.path.join(output_folder, f"{output_fname}_{shots}-shot_prompts.json"), 'w') as outfile:
      json.dump(prompts, outfile, indent=3)

   if labels!=[]:
      with my_open(os.path.join(output_folder, f"{output_fname}_{shots}-shot_labels.json"), 'w') as outfile:
         json.dump(labels, outfile, indent=3)

if __name__ == '__main__':
   fire.Fire(main)