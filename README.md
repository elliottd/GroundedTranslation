#GroundedTranslation

Dependencies
---

* python 2.7.5
* numpy 1.91
* scipy 0.15
* theano 0.7
* Keras 0.11
* h5py 2.5.0
* [dominate](https://github.com/Knio/dominate) to visualise the generated descriptions

Data
---

Download the [Flickr8K dataset](http://cs.stanford.edu/people/karpathy/deepimagesent/flickr8k.zip) from Andrej Karpathy's website. Unzip into `flickr8k`.

Training a model
---

Run `python train.py` to train a Vision-to-Language two-layer LSTM for `--epochs=50`, with `--optimiser=adagrad`, `--batch_size=100` instances, and `--l2reg=1e-8` weight regularisation. The hidden units have `--hidden_size=512` dimensions, with dropout parameters of `--dropin=0.5` and `--droph=0.2`, and an `--unk=5` threshold for pruning the word vocabulary. Training takes 500s/epoch on a Tesla K20X.

This default model should report approximately 17.7 BLEU4 on the val split on the Flickr8K dataset. 

References
---

These papers formed a basis for inspiring this project.

[Show and Tell: A Neural Image Caption Generator. Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan. CVPR '15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)

[Deep Visual-Semantic Alignments for Generating Image Descriptions. Andrej Karpathy, Li Fei-Fei. CVPR '15](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf)

[Sequence to Sequence Learning with Neural Networks. Ilya Sutskever, Oriol Vinyals, Quoc V. Le. NIPS '14.](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
