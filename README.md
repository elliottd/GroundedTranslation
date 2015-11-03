#GroundedTranslation

Dependencies
---

* python 2.7.5
* numpy 1.91
* scipy 0.15
* [Theano 0.7](https://github.com/Theano/Theano/tree/rel-0.7)
* [Keras 0.1.3](https://github.com/fchollet/keras/tree/0.1.3)
* h5py 2.5.0
* [dominate](https://github.com/Knio/dominate) to visualise the generated descriptions

Data
---

Download the [IAPRTC-12 dataset](https://www.dropbox.com/sh/xvs44ofmzs88w2b/AABzv6YmyxwXXbiBfi5AqXFKa?dl=0) from Dropbox. Unzip into `iaprtc12_eng` and `iaprtc12_ger`, respectively.

Run `python util/makejson.py --path iaprtc12_eng` followed by `python util/jsonmat2h5.py --path iaprtc12_eng` to create the dataset.h5 file expected by GroundedTranslation. Repeat this process, replacing `eng` for `ger` to create the German dataset.h5 file.

Training an English monolingual model
---

Run `THEANO_FLAGS=floatX=float32,device=gpu0 python train.py --dataset iaprtc12_eng --hidden_size=512` to train an English Vision-to-Language one-layer LSTM for `--epochs=50`, with `--optimiser=adam`, `--batch_size=100` instances, `--big_batch=10000` and `--l2reg=1e-8` weight regularisation. The hidden units have `--hidden_size=512` dimensions, with dropout parameters of `--dropin=0.5`, and an `--unk=3` threshold for pruning the word vocabulary. Training takes 500s/epoch on a Tesla K20X.

This model should report a maximum BLEU4 of 17.38 on the val split, using a fixed seed of 1234.

Training a German monolingual model
---

Run `THEANO_FLAGS=floatX=float32,device=gpu0 python train.py --dataset iaprtc12_ger --hidden_size=256` to train a German Vision-to-Language one-layer LSTM for `--epochs=50`, with `--optimiser=adam`, `--batch_size=100` instances, `--big_batch=10000` and `--l2reg=1e-8` weight regularisation. The hidden units have `--hidden_size=256` dimensions, with dropout parameters of `--dropin=0.5`, and an `--unk=3` threshold for pruning the word vocabulary. Training takes 500s/epoch on a Tesla K20X.

This model should report a maximum BLEU4 of 11.78 on the val split, using a fixed seed of 1234.

Training an English-German multilingual model
---

Run `python extract_hidden_features.py --dataset=iaprtc12_eng --checkpoint=PATH_TO_BEST_MODEL_CHECKPOINT --hidden_size=512 --h5_writeable` to extract the final hidden state representations from a saved model state. The representations will be stored in `dataset/dataset.h5` in the `final_hidden_representations` field.

Now run `THEANO_FLAGS=floatX=float32,device=gpu0 python train.py --dataset iaprtc12_ger --hidden_size=256 --source_vectors=iaprtc12_eng` to train an English Vision-to-Language one-layer LSTM for `--epochs=50`, with `--optimiser=adam`, `--batch_size=100` instances, `--big_batch=10000` and `--l2reg=1e-8` weight regularisation. The hidden units have `--hidden_size=256` dimensions, with dropout parameters of `--dropin=0.5`, and an `--unk=3` threshold for pruning the word vocabulary. Training once again takes 500s/epoch on a Tesla K20X.

This model should report a maximum BLEU4 of 13.05 on the val split, using a fixed seed of 1234. This represents a 1.28 BLEU point improvement over the German monolingual baseline.

References
---

These papers formed a basis for inspiring this project.

[Show and Tell: A Neural Image Caption Generator. Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan. CVPR '15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)

[Deep Visual-Semantic Alignments for Generating Image Descriptions. Andrej Karpathy, Li Fei-Fei. CVPR '15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf)

[Sequence to Sequence Learning with Neural Networks. Ilya Sutskever, Oriol Vinyals, Quoc V. Le. NIPS '14.](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
