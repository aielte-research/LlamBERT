from openai import OpenAI

#Not tested, too expensive, over monthly limit

client = OpenAI()

file_object = client.files.create(
  file=open("/home/projects/DeepNeurOntology/IMDB_data/ChatGPT_inputs/finetune_data/train_finetune_data.jsonl", "rb"),
  purpose="fine-tune"
)

print(client.fine_tuning.jobs.create(
  training_file=file_object.id, 
  model="gpt-3.5-turbo"
))