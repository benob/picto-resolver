import sys
import json
from datasets import Dataset, load_dataset

path = sys.argv[1]
if path.endswith('.parquet'):
  dataset = Dataset.from_parquet(path)
else:
  dataset = load_dataset(path, split='train')
batch_size = 100

from bottle import route, run, static_file, redirect

@route("/images/<filepath>")
def css(filepath):
    return static_file(filepath, root="images")

@route('/<id>')
def render(id):
  id = int(id)
  result = f'<p>[ <a href="/{int(id) - batch_size}">previous</a> - <a href="/{int(id) + batch_size}">next</a>]</p>'
  for i in range(id, id + batch_size):
    if i >= 0 and i < len(dataset):
      instance = dataset[i]
      result += f'<h3>Instance {i}</h3>'
      result += f'<p>"{instance["text"]}"</p>'
      for entry in json.loads(instance['pictos']):
        result += f'<img src="https://api.arasaac.org/v1/pictograms/{entry}?download=false" width="64"> '
      result += '<br>'
      result += f'{instance["tokens"]}'
      if 'llm_pictos' in instance:
        result += f'<details><summary>Generator details</summary><pre>{json.dumps(instance["llm_pictos"], indent=2)}</pre></details>'
      if 'resolved' in instance:
        result += f'<details><summary>Resolver details</summary><pre>{json.dumps(instance["resolved"], indent=2)}</pre></details>'
  result += '<br></p>'
  result += f'<p>[ <a href="/{int(id) - batch_size}">previous</a> - <a href="/{int(id) + batch_size}">next</a>]</p>'
  return result

@route('/')
def index():
  result = '<p>'
  for i in range(0, dataset.num_rows, batch_size):
    result += f'<a href="/{i}">{i}</a> '
  result += '</p>'
  return result

run(host='localhost', port=8080)
