# low-resource-kpgen

Official code of our work, [Representation Learning for Resource-Constrained Keyphrase Generation](https://arxiv.org/pdf/2203.08118.pdf). 

This repository contains the following functionalities:
 - Datasets used in our experiments: KP20k, Inspec, Krapivin, NUS, SemEval, KPTimes, and three 20k-document subsets of KP20k that we use as $D_{kp}$.
 - Code for intermediate representation learning and finetuning with BART
 - Code for evaluating on the benchmarks
 - Our raw predictions on the scientific benchmarks

## Citation
If you find our work useful, please consider citing:
```
@article{wu2022representation,
  title={Representation Learning for Resource-Constrained Keyphrase Generation},
  author={Wu, Di and Ahmad, Wasi Uddin and Dev, Sunipa and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:2203.08118},
  year={2022}
}
```

## Setup
- Install the requirements. We recommend installing them in a separate virtual environment as our experiments require a customized version of fairseq.
	```
	pip install -r requirements.txt
	```
- Install fairseq from source
	```
	git clone --branch model-experiment-0.10.2 https://github.com/xiaowu0162/fairseq.git
	cd fairseq
	pip install --editable ./
	cd ..
	```
- Prepare the datasets. 
	```
	cd data/scikp
	bash run.sh
	cd ../kp20k-20k
	bash run.sh
	cd ../kptimes
	bash run.sh
	```

## Finetuning on Keyphrase Generation
To finetune BART on keyphrase generation, first run preprocessing by
```
cd finetuning_fairseq
bash preprocess.sh
```
Then, training can be done by 
`bash run_train.sh GPU_IDs DATASET_NAME [BART_PATH]`
Note that 
- The hyperparameter settings are for training on kp20k-20k-1/2/3 with a single GPU. If you use more than 1 GPU, please make sure to reduce `UPDATE_FREQ` to achieve the same batch size. 
- We recommend using an effective batch size of 64 or 32 for finetuning, where 
	`effective batch size = NUM_GPUs * PER_DEVICE_BSZ * UPDATE_FREQ`
- The supported `DATASET_NAME` are `kp20k, kptimes, kp20k-20k-1, kp20k-20k-2, kp20k-20k-3`
- `BART_PATH` will default to the pre-trained BART model. You can start with other BART checkpoints (e.g., the intermediate representations pretrained with TI or SSR) by providing the corresponding `checkpoint.pt` files as parameters. To run randomly initialized BART, please remove the `--restore-file` flag in the script.

## Evaluating Keyphrase Generation
With a trained keyphrase generation model, you can run generation and evaluation by
```
bash run_test.sh GPU_ID DATASET_NAME SAVE_DIR
```
- DATASET_NAME: use `kp20k` to evaluate on all five scientific datasets.
- SAVE_DIR: the path to the checkpoint (e.g., `checkpoint_best.pt`).
- `dataset_hypotheses.txt` contains the model's raw predictions.
- `dataset_predictions.txt` contains postprocessed predictions.
- `results_log_dataset.txt` contains all the scores. 

## Learning Intermediate Representations
We provide code to run representation learning methods discussed in the paper. Note that all the hyperparameter settings are for a single GPU. We recommend running with multiple GPUs. In that case, please make sure to adjust `UPDATE_FREQ` accordingly to achieve the desired batch size.
### Text Infilling
```
cd intermediate_learning/title_generation
bash preprocess_titlegen.sh
bash run_train.sh GPU_IDs kp20k-titlegen
```
### Title Generation
```
cd intermediate_learning/text_infilling
bash preprocess_text_infilling.sh
bash run_train.sh GPU_IDs kp20k-lm
```
### Salient Span Recovery and Salient Span Prediction
SSR and SSP requires offline span prediction as the first step:
```
cd intermediate_learning/salient_span_recovery
bash find_spans_tfidf.sh
```
Then, you can run SSR by
```
bash preprocess_ssr.sh
bash run_train.sh GPU_IDs kp20k-ssr
```
Or you can run SSP by 
```
bash preprocess_ssr.sh
bash preprocess_ssp.sh
bash run_train.sh GPU_IDs kp20k-ssp
```
## Predictions from Our Models
We share the predictions from our best model [here](https://drive.google.com/file/d/1WSFgEBD7n0L55I3iB3c9beGqbGvdercv/view?usp=sharing). After downloading and uncompressing the file, you can directly run the `evaluate.sh` in the `BART-SSR-pred` folder to get the scores.
