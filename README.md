# Pictogram matcher from a lexicon of definitions and descriptions

This script matches pairs of (lemma, definition) or (lemma, description) in representation space and return the best matching pictogram according to a lexicon.

A good starting lexicon is available at https://github.com/benob/arasaac-lexicon (which contains LLM-generated descriptions and definitions for the Arasaac lexicon).
Resulting picto ids can be mapped to images with urls `https://api.arasaac.org/v1/pictograms/<picto>`.

The basic matching strategy returns the highest scoring picto according to cosine similarity in the (whitened) representation space. Representations are computed with Qwen3-Embedding-0.6B but could use any other embedding model compatible with SentenceTransformers.

A more advanced strategy requires lemma and definitions for a hypernym and a salient property that are used if the concept is not found in the lexicon. This depends on the quality of the generated definitions and descriptions.

# Dependencies

Install requirements
```
pip install -r requirements.txt
```

# Training

Download definitions.csv and descriptions.csv from https://github.com/benob/arasaac-lexicon, then
```
python resolver.py train definitions.csv descriptions.csv resolver_arasaac.pt
```

# Inference

You can perform inference from LLM-generated structured concept lists. Input instances are loaded from a parquet file and have a `llm_pictos` column which follows this format:
```
[
  {
    "concept": {
      "definition": <definition>,
      "depiction": <visual-description>,
      "lemma": <word>,
    },
    "fallback": {
      "hypernym": {
        "definition": <definition>,
        "depiction": <visual-description>, # unused at this time
        "lemma": <word>,
      },
      "salient_property": {
        "definition": <definition>,
        "depiction": <visual-description>, # unused at this time
        "lemma": <word>,
      }
    }
  },
  ...
]
```

Each concept is matched according to the following rules:
- choose the visual picto if its score is higher than that of the lexical picto
- else choose the lexical picto if score > 0.3 or score > mean hypernym and property pictos
- else choose both hypernym and property pictos if property score > 0.3
- else choose hypernym
Consecutive duplicates and pictos with score < 0.2 are then filtered out

The result is stored in columns "pictos" (json string representing the list of pictos), "tokens" (first lemma of selected pictos), "resolved" (detailed resolver output)

```
python resolver.py inference resolver_arasaac.pt common-voice_generated.parquet common-voice_resolved.parquet
```

# API

## Training

Compute embeddings for the lexicon and the visual descriptions, store everything in pt file. Device can be cuda, mps, cpu, and is auto-selected if not specified.
```
train(lexicon_path: str, visual_lexicon_path: str, output_path: str, model_name="Qwen/Qwen3-Embedding-0.6B", device: str=None)
```

## Inference
Instanciate a trained model
```
resolver = Resolver("path-to-model.pt", device=None)
```

### Simple Inference:

Get the highest scoring picto for a list of concepts using either the "lexical" or "visual" embedding. Concepts are expected to be dicts with keys "lemma", "definition" and "depiction". An optional `kind` can be specified to populate the source of the result.

Results are instances of 

```
@dataclass
class Result:
  picto: str            # picto id
  score: float          # match score
  lemma: str            # list of matched lemmas
  definition: str       # matched definition or description
  input_lemma: str      # input word
  input_definition: str # input definition
  kind: str             # copy of the kind parameter 
```

A custom getter can be specified for extracting each concept from the input list, as well as custom keys for indexing the result of the getter.
```
resolver.inference(concepts, embedding, getter=lambda x: x, lemma_key='lemma', definition_key='definition', depiction_key='depiction', kind=None)
```

### Inferance with fallback strategy

Perform inference with multiple strategies to handle OOV or low-scoring concepts.
```
resolver.inference_with_fallback(sentence)
```

This assumes that each concept is a dict with the following fields
```
  {
    "concept": {
      "definition": <definition>,
      "depiction": <visual-description>,
      "lemma": <word>,
    },
    "fallback": {
      "hypernym": {
        "definition": <definition>,
        "depiction": <visual-description>, # unused at this time
        "lemma": <word>,
      },
      "salient_property": {
        "definition": <definition>,
        "depiction": <visual-description>, # unused at this time
        "lemma": <word>,
      }
    }
  }
```

The matching strategy is specified earlier in the command-line utility documentation.

### Embedding matcher

A low-level embedding matcher can be used. It whitens the embedding space and returns the highest cosine similarity results in that space. 
```
matcher.batch_inference(lemmas: List[str], definitions: List[str], batch_size=64, kind=None) -> list[Result]
```
Batch inference performs lookups of a set of lemmas, which is faster than individual matches.

### Post processing

Simply removes consecutive duplicates and pictos with a score bellow a threshold.
```
resolver.post_process(results, threshold=0.2, remove_dupes=True)
```
