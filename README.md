# Image Captioning with Concept
<b> This project is based on soft-attention mechanism, which is applied on both spacial feature map extracted from the image by Resnet-101 and semantic features, which are most relevant tags extracted by YOLO-9000.

## Dependencies:

- Python 3.5
- pytorch 1.0.0
- torchvision
- tensorboardX
- skimage
- tqdm
- [coco-caption](https://github.com/tylin/coco-caption). There is some errors appear when using this repo along with our code due to confliction of Python versions, please check their issues tab or email us if needed.
- [Yolo-9000](https://github.com/philipperemy/yolo-9000.git)
- Some other small libraries haven't been listed, please feel free to install it if being required while running code.

## Getting Started:

Before training or inference, you have to run Yolo-9000 on the directory containing images you want to use, the original code cannot be run automatically to extract tags from multiple images and save them into a file, please feel free to email us if you want to reproduce the results, we will upload our light additional source code we made to extract tags on multiple images and a tutorial for it when we have enough time.

### Training:

In order to train the model, you need to download the dataset's images and annotations, you can mannually download them from [MSCOCO's site](http://cocodataset.org) or run below bash file to download training, validation and testing sets:

```ruby
$ bash download.sh
```

We used the 2017 version's training and validation sets because they are splitted following the recommendations of some prior works, but if you want to submit the results to [evaluation site](https://competitions.codalab.org/competitions/3221), your model have to do inference on 2014 version's validation and testing sets.

After having the dataset prepared, preprocess the images and captions and then run training code, you're free to set configuration of the training phrase by modifying [train.py](train.py) file:

```ruby
$ python prepro.py
$ python train.py
```

Make sure that you have enough space in your drive, it would take about **130GB** after preprocessing because the extracted features are big.

While training, you can observe the process by tensorboard:

```ruby
$ tensorboard --logdir=log/
```

### Evaluation and Inference:

Run the below line to to get help in inference, if you want results to be evaluated after inference (only if there are evaluation annotations), set the --split argument to val:

```ruby
$ python infer.py --help
```

## References:

The code we used as the reference while building our code (we want to acknowledge the authors of this repo because we have borrowed some ideas from this repo on the first build that is the basement of this source code): https://github.com/yunjey/show-attend-and-tell
