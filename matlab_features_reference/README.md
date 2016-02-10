This directory contains the files used to extract the visual features for the WMT16 Shared Task.

We used code from Andrej Karpathy's [neuraltalk](https://github.com/karpathy/neuraltalk).

We modified the [original VGG-19 deploy.protoxt](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-vgg_ilsvrc_19_layers_deploy-prototxt) to extract features from the 'relu7' and 'conv5_4' layers. See the following files for details.

* deploy_features-conv54.prototxt (extract CONV5_4 features)
* deploy_features-fc7.prototxt (extract FC_7 features)
