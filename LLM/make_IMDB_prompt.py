import os
import json
from typing import Optional
import fire

import sys
sys.path.append("..")
from utils import my_open

def get_sys_prompt_plain():
   return "Please answer with 'positive' or 'negative' only!"

def get_sys_prompt():
   return {
      "role": "system",
      "content": get_sys_prompt_plain()
   }

def get_user_prompt_plain(rev_str):
   return f"Decide if the following movie review is positive or negative: \n{rev_str}\n If the movie review is positive please answer 'positive', if the movie review is negative please answer 'negative'. Make your decision based on the whole text."

def get_user_prompt(rev_str):
   return {
      "role": "user",
      "content": get_user_prompt_plain(rev_str)
   }

def get_assistant_prompt(content):
   return {
      "role": "assistant",
      "content": content
   }


def main(
   input_fpath: str="/home/projects/DeepNeurOntology/IMDB_data/promt_eng.json",
   output_folder: str="model_inputs/IMDB/",
   output_fname: Optional[str]=None,
   shots: int=0,
   plain_format: bool=False,
   openai_format: bool=False,
   **kwargs
):
   if len(kwargs) > 0:
      raise ValueError(f"Unknown argument(s): {kwargs}")
   with open(input_fpath) as f:
      reviews = json.loads(f.read())

   if output_fname is None:
      output_fname = os.path.basename(input_fpath).split(".")[0]

   rev_str_1 = "I don't know where to start; the acting, the special effects and the writing are all about as bad as you can possibly imagine. I can't believe that the production staff reached a point where they said, \"Our job is done, time for it's release\". I'm just glad the first two in the series never made it as far as the UK. I would actually recommend watching this film just so you can appreciate how well made most films are.<br /><br />I don't know how any of the other IMDb users could find it scary when the \"terrifying\" dinosaurs waddle down corridors with rubber arms flailing around."
   rev_str_2 = "Everybody knows that Gregory Widen's original \"The Prophecy\" didn't really require a sequel, but you also don't need a degree in rocket science hanging above your chimney to realize that further cash-ins on this profitable horror concept were inevitable. Part two is a very prototypic example of a straight-to-video sequel, meaning the creative and convoluted plot of the original has been simplified a lot in favor of more action, more witty one-liners and a lot more eerie religious scenery. The only good news is that the producers managed to keep Christopher Walken for the role of Gabriel, and he delivers another gloriously brazen performance that promptly justifies the price of a rental. If it wasn't for Walken's performance (and perhaps a couple of players in the supportive cast like Brittany Murphy and Glenn Danzig), \"The Prophecy II\" surely would have disappeared into oblivion straight after its release. The movie begins with Gabriel literally getting spat out of hell to proceed with his ongoing War of Heaven here on earth. The purpose of his battle this time is to prevent the baby of nurse Valerie Rosales (Jennifer Beals) from getting born. For you see, her unborn child is the first ever hybrid between a heavenly angel and an earthly \"monkey\" and the birth of such a superior being would imply the downfall of Gabriel's evil dominion. Thus, just as in the first movie, he engages a suicidal accomplice to assist him and hunts Valerie all the way down to the Eden for the final showdown. \"The Prophecy II\" is an endurable and occasionally even entertaining movie as long as you don't make comparisons with the original and as long as you manage to overlook the multiple plot holes and errors in continuity. Whenever the storyline becomes too tedious, the makers luckily enough always insert a near-brilliant Christopher Walken moment to distract you. His interactions with the rebellious Izzy and particularly his ignorance regarding modern earthly technologies often result in worthwhile and memorable sequences. On a slightly off-topic note, I often felt like \"The Prophecy II\" ambitions to look similar to \"Terminator II\" \u0085 Gabriel's resurrection looked somewhat like the teleportation of a futuristic cyborg and the Eden location, where the final battle takes place, looks very similar to the steel factory where \"Terminator II\" ended as well. Coincidence, I guess? Overall, this is an inferior and passable sequel but still worth checking out in case you're a fan of Christopher Walken's unique acting charisma (and who isn't?)"
   rev_str_3 = "As much as I have enjoyed the Hanzo the Razor movies, three is definitely enough: 'Who's Got The Gold?', the final adventure for the Japanese lawman with the impressive package, is a fairly enjoyable piece of Pinku cinema, but offers little new in terms of ideas whilst taking a big step backwards as far as outrageousness is concerned.<br /><br />The film opens with the appearance of a female ghost, and looks as though it is going to explore supernatural territory, something which might have taken the series in an interesting new direction; unfortunately, after the spook turns out to be nothing but a Scooby Doo-style ruse (cooked up by a corrupt treasury official keen to keep people away from the lake where he is hiding stolen gold), director Yoshio Inoue is content to recycle familiar elements from the first two films, the result being a rather stale affair.<br /><br />Once again, Hanzo heads an investigation that requires him to interrogate women through the use of his mighty penis, slice up his enemies, and abuse his superiors. On the way, we get wild orgies, good-natured rape (Hanzo forces himself on women who wind up appreciating his willfulness), and bloody sword-fights.<br /><br />If you've already seen and appreciated the first two films, you might as well watch this instalment to complete the set, but be warned, this is probably the least satisfying one of them all.<br /><br />6.5 out of 10, rounded up to 7 for IMDb."
   
   prompts=[]
   labels=[]   

   for rev in reviews:
      if isinstance(rev, dict):
         rev_str=rev["txt"]
         if "label" in rev:
            labels.append(rev["label"])
      elif isinstance(rev, str):
         rev_str=rev
      else:
         raise ValueError(f"Unknown input type: {type(rev)}")
      
      if plain_format:
         prompts.append(get_user_prompt_plain(rev_str))
         
      elif openai_format:         
         prompt = []
         if shots>=1:
            prompt.append(get_sys_prompt())
            prompt.append(get_user_prompt(rev_str_1))
            prompt.append(get_assistant_prompt("negative"))
         if shots>=2:
            prompt.append(get_sys_prompt())
            prompt.append(get_user_prompt(rev_str_2))
            prompt.append(get_assistant_prompt("negative"))
         if shots>=3:
            prompt.append(get_sys_prompt())
            prompt.append(get_user_prompt(rev_str_3))
            prompt.append(get_assistant_prompt("positive"))
         if shots>=4:
            raise ValueError("Only options 0, 1, 2 and 3 shot are implemented.")
         prompt.append(get_sys_prompt())
         prompt.append(get_user_prompt(rev_str))
         prompts.append(prompt)

      else: #Llama format
         prompt = ""

         if shots>=1:
            prompt += f"[INST] <<SYS>>\n{get_sys_prompt_plain()}\n<</SYS>>\n {get_user_prompt_plain(rev_str_1)} [/INST]\nnegative\n"
         if shots>=2:
            prompt += f"[INST] <<SYS>>\n{get_sys_prompt_plain()}\n<</SYS>>\n {get_user_prompt_plain(rev_str_2)} [/INST]\nnegative\n"
         if shots>=3:
            prompt += f"[INST] <<SYS>>\n{get_sys_prompt_plain()}\n<</SYS>>\n {get_user_prompt_plain(rev_str_3)} [/INST]\npositive\n"
         if shots>=4:
            raise ValueError("Only options 0, 1, 2 and 3 shot are implemented.")
         prompt += f"[INST] <<SYS>>\n{get_sys_prompt_plain()}\n<</SYS>>\n {get_user_prompt_plain(rev_str)} [/INST]\n"
      
      prompts.append(prompt)
   
   with my_open(os.path.join(output_folder, f"{output_fname}_{shots}-shot_prompts.json"), 'w') as outfile:
      json.dump(prompts, outfile, indent=3)

   if labels!=[]:
      with my_open(os.path.join(output_folder, f"{output_fname}_{shots}-shot_labels.json"), 'w') as outfile:
         json.dump(labels, outfile, indent=3)

if __name__ == '__main__':
   fire.Fire(main)