# Fine-Tuning LLMs for Cross-Domain Sequential Recommendation: A Reproducibility Study of the LLMCDSR Framework
This is the official implementation of the "Fine-Tuning LLMs for Cross-Domain Sequential Recommendation: A Reproducibility Study of the LLMCDSR Framework"

## Requirements
- torch == 2.0.1
- transformers == 4.31.0
- higher == 0.2.1
- lora 

## Data
The processed data used in our work (i.e., music-games and pet-beauty) are in `./data`.

It is taken from the amazon review dataset (2023): https://amazon-reviews-2023.github.io

## Download meta-llama/Meta-Llama-3-8B-Instruct from hugging face


## Candidate-Free Cross-Domain Interaction Generation

*One can use the ready-made textual embeddings downloaded above to skip this.*
Note: Before running every script, make sure you load the right data paths and apply the right weights to the model
1. Run to get the generations of LLMs.
```shell
cd ./generation
python candidate_generate_icl.py --base_model={LLM_model_path} --output_name={generation_dir_name}
```
2. As generations may not be standard listed results, use LLM again to parse the names of the generated interaction items:
```shell
python parse_items.py --base_model={LLM_model_path} --data_name={generation_dir_name}
```
3. First, get the textual embeddings of the items in the dataset.
```shell
cd ./generation
python get_item_embedding.py --task={dataset} --domain={A|B} --model_path={text_embedding_model_path}
```
4. Get the textual embeddings of the parsed generations:
```shell
python get_candidate_embeddings.py --model_path={text_embedding_model_path} --task={dataset} --generation_path={parsed_generation_path}
```

## Collaborative-Textual Contrastive Pre-Training
Make sure you have the right data paths and weights on the model

Run the following commands to pre-train.
```shell
cd ./pre-train
python main.py --dataset={dataset} --domain={A|B}
```
After that, copy the trained parameters into `./pretrained_parameters` fold with the name of `{dataset}_projection_{A|B}.pt`

## Relevance-Aware Meta Recall Network
Once having prepared the needed ingradients, one can simply run the code to train the model and evaluate the performance:
```shell
python main.py --dataset={dataset}
```

## Fine tuning a model:
1. Run to get the generations of LLMs. Use already generated files in data; "music_to_games.jsonl" and "pet_to_beauty.jsonl
```shell
cd ./finetune
python train_lora.py
```
2. Apply the weights to the model generation/candidate_generate_icl.py 

