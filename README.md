# GroVE

## Dataset
The code supports the MS-COCO, Flickr30k, CUB-200-2011 and Oxford Flowers 102.

COCO: Download the 2014 data containing [images and captions](https://cocodataset.org/#home) and setup the directory in the following way

```bash
coco
|-images/
|--train2014 
|--val2014 
|-captions_train2014.json 
|-captions_val2014.json
```
Flickr30k: Download the [images and captions](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset). 
CUB-200-2011: Download the CUB-200-2011 [images](http://www.vision.caltech.edu/datasets/cub_200_2011/) and the [captions](https://drive.google.com/file/d/1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) 
Oxford Flowers 102: Download the [images](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) and [captions](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view?resourcekey=0-Av8zFbeDDvNcF1sSjDR32w)

## Training and Evaluation
Update the `config.py` script with the dataset path and the hyperparameter values. 
Run the training using `train.py`. For evaluation, use `evaluation.py`.
