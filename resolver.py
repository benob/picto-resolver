from collections import defaultdict
from dataclasses import dataclass, asdict
import sys
import itertools
import csv
import json
from typing import Dict, List, Tuple

from tqdm import tqdm
from unidecode import unidecode
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import numpy as np

def normalize_text(lemma):
  return unidecode(lemma).lower().strip()

def prepare_text(lemma, definition):
  return unidecode(lemma).lower().strip() + ': ' + definition

def train(lexicon_path: str, visual_lexicon_path: str, output_path: str, model_name="Qwen/Qwen3-Embedding-0.6B", device: str=None):
  #print('LEXICON', lexicon_path, file=sys.stderr)
  #print('VISUAL', visual_lexicon_path, file=sys.stderr)
  #print('OUTPUT_PATH', output_path, file=sys.stderr)
  if device is None:
    device = next(filter(lambda name: getattr(torch, name).is_available(), ['cuda', 'mps', 'cpu']))
    print('DEVICE', device, file=sys.stderr)

  model = SentenceTransformer(model_name)
  model.to(device)

  ordered_descriptions = []
  descriptions = {}
  with open(visual_lexicon_path) as fp:
    reader = csv.reader(fp, delimiter='\t')
    #next(reader)
    for id, text in reader:
      descriptions[str(id)] = text

  picto = []
  definitions = []
  words = []
  lexicon = defaultdict(list)
  reverse = defaultdict(list)

  with open(lexicon_path) as fp:
    reader = csv.reader(fp, delimiter='\t')
    #next(reader)
    for id, lemmas, definition in reader:
      #print(id)
      lemmas = json.loads(lemmas)
      definition = lemmas[0] + ': ' + definition
      reverse[id].append(len(definitions))
      definitions.append(definition)
      ordered_descriptions.append(lemmas[0] + ': ' + descriptions[id])
      picto.append(id)
      for lemma in lemmas:
        lexicon[normalize_text(lemma)].append([len(definitions), id, lemmas, definition])
      words.append(lemmas)

  lexicon = dict(lexicon)
  reverse = dict(reverse)

  batch_size = 64

  embeddings = []
  for batch in itertools.batched(tqdm(definitions), batch_size):
    embeddings.extend(model.encode(batch))
  embeddings = torch.tensor(np.array(embeddings))
  print('LEXICAL', embeddings.shape, file=sys.stderr)

  visual_embeddings = []
  for batch in itertools.batched(tqdm(ordered_descriptions), batch_size):
    visual_embeddings.extend(model.encode(batch))
  visual_embeddings = torch.tensor(np.array(visual_embeddings))
  print('VISUAL', visual_embeddings.shape, file=sys.stderr)

  torch.save({'model_name': model_name, 'picto': picto, 'definitions': definitions, 'words': words, 'lexicon': lexicon, 'embeddings': embeddings, 'reverse': reverse, 'visual_embeddings': visual_embeddings, 'visual_descriptions': ordered_descriptions}, output_path)

@dataclass
class Result:
  picto: str
  score: float
  lemma: str
  definition: str
  input_lemma: str
  input_definition: str
  kind: str
    

class EmbeddingMatcher:
  def __init__(self, encoder, embeddings: torch.Tensor, pictos: List[str], lemmas: List[str], definitions: List[str], eps=1e-5):
    mean = embeddings.mean(dim=0, keepdim=True)
    X = embeddings - mean
    cov = (X.T @ X) / X.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    inv_sqrt = torch.diag(torch.rsqrt(eigvals + eps))
    self.encoder = encoder
    self.whitening = eigvecs @ inv_sqrt @ eigvecs.T
    self.mean = mean.squeeze(0)
    self.embeddings = self.whiten(embeddings)
    self.pictos = pictos
    self.lemmas = lemmas
    self.definitions = definitions

  def whiten(self, embeddings: torch.Tensor) -> torch.Tensor:
    X = embeddings - self.mean
    return X @ self.whitening

  def similarity(self, x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == self.embeddings.shape[-1], "input size different from that of embeddings"

    xw = self.whiten(x)
    return F.cosine_similarity(self.embeddings, xw, dim=-1)

  def batch_inference(self, lemmas: List[str], definitions: List[str], batch_size=64, kind=None) -> list[Result]:
    assert len(lemmas) == len(definitions), "same number of lemmas and definitions required"

    prepared_definitions = [prepare_text(lemma, definition) for lemma, definition in zip(lemmas, definitions)]
    results = []

    for i in range(0, len(prepared_definitions), batch_size):
      batch = prepared_definitions[i: i + batch_size]
      embeddings = self.encoder.encode(batch)
      scores = self.similarity(torch.tensor(embeddings).unsqueeze(1))
      best = torch.argmax(scores, dim=-1)

      for rank, found in enumerate(best):
        results.append( 
          Result(self.pictos[found], scores[rank][found].item(), self.lemmas[found], self.definitions[found], lemmas[i + rank], definitions[i + rank], kind)
        )

    return results

  def inference(self, lemma: str, definition: str, kind=None) -> Result:
    return self.batch_inference([lemma], [definition], kind=kind)[0]
    

class Resolver:
  def __init__(self, model_path, device=None):
    if device is None:
      device = next(filter(lambda name: getattr(torch, name).is_available(), ['cuda', 'mps', 'cpu']))
      print('DEVICE', device, file=sys.stderr)
    self.model_path = model_path
    data = torch.load(model_path, weights_only=False)

    self.model_name, self.picto, self.words, self.lexicon, self.reverse = data['model_name'], data['picto'], data['words'], data['lexicon'], data['reverse']

    self.lexical_definitions, lexical_embeddings = data['definitions'], data['embeddings']
    self.visual_descriptions, visual_embeddings = data['visual_descriptions'], data['visual_embeddings']

    self.model = SentenceTransformer(self.model_name)
    self.model.to(device)

    self.lexical_embeddings = EmbeddingMatcher(self.model, lexical_embeddings, self.picto, self.words, self.lexical_definitions)
    self.visual_embeddings = EmbeddingMatcher(self.model, visual_embeddings, self.picto, self.words, self.visual_descriptions)

  def inference(self, concepts, embedding, getter=lambda x: x, lemma_key='lemma', definition_key='definition', depiction_key='depiction', kind=None):
    lemmas = [getter(x)[lemma_key] for x in concepts]
    if embedding == 'lexical':
      definitions = [getter(x)[definition_key] for x in concepts]
      return self.lexical_embeddings.batch_inference(lemmas, definitions, kind=kind)
    elif embedding == 'visual':
      depictions = [getter(x)[depiction_key] for x in concepts]
      return self.visual_embeddings.batch_inference(lemmas, depictions, kind=kind)
    else:
      raise NotImplemented

  def inference_with_fallback(self, sentence):
    concepts = self.inference(sentence, 'lexical', getter=lambda x: x['concept'], kind='concept')
    hypernyms = self.inference(sentence, 'lexical', getter=lambda x: x['fallback']['hypernym'], kind='hypernym')
    attributes = self.inference(sentence, 'lexical', getter=lambda x: x['fallback']['salient_property'], kind='attribute')
    visual_matches = self.inference(sentence, 'visual', getter=lambda x: x['concept'], kind='visual')
    result = []
    for concept, visual, (hypernym, attribute) in zip(concepts, visual_matches, zip(hypernyms, attributes)):
      #print(concept.input_lemma, concept.score, visual.score, hypernym.score, attribute.score)
      if visual.score > concept.score:
        result.append(visual)
      elif concept.score > 0.3 or concept.score > (hypernym.score + attribute.score) / 2:
        result.append(concept)
      elif attribute.score > 0.3:
        result.append(hypernym)
        result.append(attribute)
      else:
        result.append(hypernym)
    return self.post_process(result)

  def post_process(self, results, threshold=0.2, remove_dupes=True):
    filtered = []
    for i, item in enumerate(results):
      if item.score >= threshold and (not remove_dupes or i == 0 or results[i - 1].picto != item.picto):
        filtered.append(item)
    return filtered

if __name__ == '__main__':
  def usage():
      print(f'USAGE: {sys.argv[0]} train <definitions.csv> <descriptions.csv> <model.pt>', file=sys.stderr)
      print(f'       {sys.argv[0]} inference <model.pt> <parquet>', file=sys.stderr)
      sys.exit(1)

  if len(sys.argv) == 5 and sys.argv[1] == 'train':
    train(sys.argv[2], sys.argv[3], sys.argv[4])

  elif len(sys.argv) == 5 and sys.argv[1] == 'inference':
    from datasets import Dataset
    from tqdm import tqdm

    resolver = Resolver(sys.argv[2])
    dataset = Dataset.from_parquet(sys.argv[3])
    
    new_instances = []
    for instance in tqdm(dataset):
      #result = resolver.inference(instance['llm_pictos'], kind='visual', getter=lambda x: x['concept'])
      result = resolver.inference_with_fallback(instance['llm_pictos'])
      instance['pictos'] = json.dumps([x.picto for x in result])
      instance['tokens'] = ' '.join([x.lemma[0].replace(' ', '_') for x in result])
      instance['resolved'] = [asdict(x) for x in result]
      new_instances.append(instance)

    Dataset.from_list(new_instances).to_parquet(sys.argv[4])

  elif len(sys.argv) == 2:
    resolver = Resolver(sys.argv[1])
    sentence = []
    for line in sys.stdin:
      if ':' in line:
        lemma, definition = line.strip().split(':', 2)
      else:
        lemma, definition = line.strip(), ''
      sentence.append({'lemma': lemma, 'definition': definition})

    for result in resolver.inference(sentence):
      print(result)

  else:
    usage()


