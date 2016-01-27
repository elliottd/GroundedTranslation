#GroundedTranslation

![Alt text](https://staff.fnwi.uva.nl/d.elliott/GroundedTranslation/overview.png?raw=true "Optional Title")

This is the source code that accompanies [Multilingual Image Description with Neural Sequence Models](http://arxiv.org/abs/1510.04709) . You can use it to train multilingual multimodal language models for image description.

Dependencies
---

* CUDA 6.5.14
* python 2.7.5
* numpy 1.91
* scipy 0.15
* [Theano 0.7](https://github.com/Theano/Theano/tree/rel-0.7)
* [Keras 0.1.3](https://github.com/fchollet/keras/tree/0.1.3)
* h5py 2.5.0
* [dominate](https://github.com/Knio/dominate) to visualise the generated descriptions

Data
---

Download a pre-processed version of the IAPRTC-12 dataset for [English](https://www.dropbox.com/s/lmpjbbozuaebisj/eng.tar.gz) and [German](https://www.dropbox.com/s/u6d9tt88ncst5da/ger.tar.gz) from Dropbox. Unzip into `iaprtc12_eng` and `iaprtc12_ger`, respectively.

Run `python util/makejson.py --path iaprtc12_eng` followed by `python util/jsonmat2h5.py --path iaprtc12_eng` to create the dataset.h5 file expected by GroundedTranslation. Repeat this process, replacing `eng` for `ger` to create the German dataset.h5 file.

Training an English monolingual model
---

Run `THEANO_FLAGS=floatX=float32,device=gpu0 python train.py --dataset iaprtc12_eng --hidden_size=256 --fixed_seed --run_string=fixed_seed-eng256mlm` to train an English Vision-to-Language one-layer LSTM. Training takes 500s/epoch on a Tesla K20X.

By default, this uses `--optimiser=adam`, `--batch_size=100` instances, `--big_batch=10000` and `--l2reg=1e-8` weight regularisation. The hidden units have `--hidden_size=256` dimensions, with dropout parameters of `--dropin=0.5`, and an `--unk=3` threshold for pruning the word vocabulary. 

This model should report a maximum BLEU4 of 15.21 (PPLX 6.898) on the val split, using a fixed seed of 1234.

Training a German monolingual model
---

Run `THEANO_FLAGS=floatX=float32,device=gpu0 python train.py --dataset iaprtc12_ger --hidden_size=256  --fixed_seed --run_string=fixed_seed-ger256mlm` to train a German Vision-to-Language one-layer LSTM. Training takes 500s/epoch on a Tesla K20X.

By default, this uses `--optimiser=adam`, `--batch_size=100` instances, `--big_batch=10000` and `--l2reg=1e-8` weight regularisation. The hidden units have `--hidden_size=256` dimensions, with dropout parameters of `--dropin=0.5`, and an `--unk=3` threshold for pruning the word vocabulary. 

This model should report a maximum BLEU4 of 11.91 (PPLX 9.347) on the val split, using a fixed seed of 1234.

Extracting Hidden Features from a Trained Model
---

Run `THEANO_FLAGS=floatX=float32,device=gpu0 python extract_hidden_features.py --dataset=iaprtc12_eng --model_checkpoints=PATH_TO_MODEL_CHECKPOINTS --hidden_size=256 --h5_writeable` to extract the final hidden state representations from a saved model state. The representations will be stored in `dataset/dataset.h5` in the `gold-hidden_feats-vis_enc-256` field.

You can add `--use_predicted_tokens`, `--hidden_size`, and `--no_image` to affect the label of the storage field. Specifically, `--hidden_size` can only be varied with an appropriately trained model. `--no_image` can only be varied with a model trained over only word inputs. `--use_predicted_tokens` only makes sense with an MLM.

* `--hidden_size=512` -> `gold-hidden_feats-vis_enc-512` (multimodal hidden features with 512 dims)
* `--use_predicted_tokens` -> `predicted-hidden_feats-vis_enc-256` (hidden features from *predicted* descriptions)
* `--no_image` -> `gold-hidden_feats-mt_enc-256` (LM-only hidden features)

Training Multilingual Multimodal Models
---

If you want to train a German model with transferred features from English, run `THEANO_FLAGS=floatX=float32,device=gpu0 python train.py --dataset iaprtc12_ger --hidden_size=256  --fixed_seed --source_vectors=iaprtc12_eng --source_type=gold --source_enc=vis_enc --run_string=fixed_seed-eng256mlm-ger256mlm` to train a German-to-English one-layer LSTM.

By default, this uses `--optimiser=adam`, `--batch_size=100` instances, `--big_batch=10000` and `--l2reg=1e-8` weight regularisation. The hidden units have `--hidden_size=256` dimensions, with dropout parameters of `--dropin=0.5`, and an `--unk=3` threshold for pruning the word vocabulary.

This model should report a maximum BLEU4 of 14.79 (PPLX 9.525) on the val split, using a fixed seed of 1234. This represents a 2.88 BLEU point improvement over the German monolingual baseline.

In the other direction, let's train an English model with transferred German features: `THEANO_FLAGS=floatX=float32,device=gpu0 python train.py --dataset iaprtc12_eng --hidden_size=256  --fixed_seed --source_vectors=iaprtc12_ger --source_type=gold --source_enc=vis_enc --run_string=fixed_seed-ger256mlm-eng256mlm`. This model should report a maximum BLEU4 of 19.78 (PPLX 6.148) on the val split, using a fixed seed of 1234. This represents a 4.57 BLEU point improvement over the monolingual baseline.

References
---

[Multilingual Image Description with Neural Sequence Models. Desmond Elliott, Stella Frank, Eva Hasler.](http://arxiv.org/abs/1510.04709)
