[![PyPI version](https://badge.fury.io/py/motmetrics.svg)](https://pypi.org/project/ivtmetrics/0.0.1/)

# ivtmetrics

The **ivtmetrics** library provides a Python implementation of metrics for benchmarking surgical action triplet detection and recognition.

## Features at a glance

The following are available with ivtmetrics:

1. **Recognition Evaluation**: Provides AP metrics to measure the performance of a model on action triplet recognition. 
2. **Detection Evaluation**: Supports Intersection over Union distances measure of the triplet localization with respect to the instruments.
3. **Flexible Analysis**: (1) Supports for switching between frame-wise to video-wise averaging of the AP.
(2) Supports disentangle prediction and obtained filtered performance for the various components of the triplets as well as their association performances at various levels. 



<a name="installation"></a>
## Installation

### Install via PyPi

To install **ivtmetrics** use `pip`
```
pip install ivtmetrics
```

### Install via Conda

```
conda install -c nwoye ivtmetrics
```

Python 3.5-3.9 and numpy and scikit-learn are required.

<a name="Metrics"></a>
## Metrics

The metrics have been aligned with what is reported by [CholecT50](https://www.sciencedirect.com/science/article/abs/pii/S1361841522000846) benchmark.
**ivtmetrics** can be imported in the following way: 

``` python 
import ivtmetrics
```

The metrics implement both **recognition** and **detection** evaluation.
The metrics internally implement a disentangle function to help filter the triplet components as well as triplet different levels of association.

### Recognition Metrics

**Recognition ivtmetrics** can be used in the following ways:

``` python
metric = ivtmetrics.Recognition(num_class)
```
This takes an argument `num_class` which is default to `100`

The following function are possible with the `Recognition` class:

Name | Description
:--- | :---
update(`targets, predictions`)|takes in a (batch of) vector predictions and their corresponding groundtruth. vector size must match `num_class` in the class initialization.
video_end()|Call to make the end of one video sequence.
reset()|Reset current records. Useful during training and can be called at the begining of each epoch to avoid overlapping epoch performances.
reset_global()|Reset all records. Useful for switching between training/validation/testing or can be called at the begining of new experiment.
compute_AP(`component, ignore_null`)|Obtain the average precision on the fly. This gives the AP only on examples cases after the last `reset()` call. Useful for epoch performance during training. 
compute_video_AP(`component, ignore_null`)|(RECOMMENDED) compute video-wise AP performance as used in CholecT50 benchmarks.
compute_global_AP(`component, ignore_null`)|compute frame-wise AP performance for all seen samples.
topK(`k, component`) | Obtain top K performance on action triplet recognition for all seen examples. args `k` can be any int between 1-99. k = [5,10,15,20] have been used in benchmark papers.
topClass(`k, component`)|Obtain top K recognized classes on action triplet recognition for all seen examples. args `k` can be any int between 1-99. k = 10 have been used in benchmark papers.

### args:
- args `component` can be any of the following ('i', 'v', 't', 'iv', 'it','ivt') to compute performance for (instrument, verb, target, instrument-verb, instrument-target, instrument-verb-target) respectively. default is 'ivt' for triplets.
- args `ignore_null` (optional, default=False): to ignore null triplet classes in the evaluation. This option is enabled in CholecTriplet2021 challenge.
- the output is a `dict` with keys("AP", "mAP") for per-class and mean AP respectively.






<a name="recognitionExample"></a>
#### Example usage



```python
import ivtmetrics
recognize = ivtmetrics.Recognition(num_class=100)
network = MyModel(...) # your model here 
# training
for epoch in number-of-epochs:
  recognize.reset()
  for images, labels in dataloader(...): # your data loader
    predictions = network(image)
    recognize.update(labels, predictions)
  results_i = recognize.compute_AP('i')
  print("instrument per class AP", results_i["AP"])
  print("instrument mean AP", results_i["mAP"])
  results_ivt = recognize.compute_AP('ivt')
  print("triplet mean AP", results_ivt["mAP"])

# evaluation
recognize.reset_global()
for video in videos:
  for images, labels in dataloader(video, ..): # your data loader
    predictions = network(image)
    recognize.update(labels, predictions)
  recognize.video_end()
    
results_i = recognize.compute_video_AP('i')
print("instrument per class AP", results_i["AP"])
print("instrument mean AP", results_i["mAP"])

results_it = recognize.compute_video_AP('it')
print("instrument-target mean AP", results_it["mAP"])

results_ivt = recognize.compute_video_AP('ivt')
print("triplet mean AP", results_ivt["mAP"])
```

Any `nan` value in results is for classes with no occurrence in the data sample.





### Detection Metrics

**Detection ivtmetrics** can be used in the following ways:

```python
metric = ivtmetrics.Detection(num_class, num_tool, threshold=0.5)

```
This takes an argument `num_class` which is default to `100` and `num_tool` which is default to `6`

The following function are possible with the `Detection` class:

Name | Description
:--- | :---
update(`targets, predictions, format`)|input: takes in a (batch of) list/dict predictions and their corresponding groundtruth. Each frame prediction/groundtruth can be either as a `list of list` or `list of dict`. (more details below).
video_end()|Call to make the end of one video sequence.
reset()|Reset current records. Useful during training and can be called at the begining of each epoch to avoid overlapping epoch performances.
reset_global()|Reset all records. Useful for switching between training/validation/testing or can be called at the begining of new experiment.
compute_AP(`component`)|Obtain the average precision on the fly. This gives the AP only on examples cases after the last `reset()` call. Useful for epoch performance during training.
compute_video_AP(`component`)|(RECOMMENDED) compute video-wise AP performance as used in CholecT50 benchmarks.
compute_global_AP(`component`)|compute frame-wise AP performance for all seen samples.

### args:
1. **list of list format**: [[tripletID, toolID, toolProbs, x, y, w, h], [tripletID, toolID, toolProbs, x, y, w, h], ...], where: 
    * `tripletID` = triplet unique identity
    * `toolID` = instrument unique identity
    * `toolProbs` = instrument detection confidence
    * `x` = bounding box x1 coordiante
    * `y` = bounding box y1 coordinate
    * `w` = width of the box
    * `h` = height of the box
    * The [x,y,w,h] are scaled between 0..1

2. **list of dict format**: [{"triplet":tripletID, "instrument":[toolID, toolProbs, x, y, w, h]}, {"triplet":tripletID, "instrument":[toolID, toolProbs, x, y, w, h]}, ...]. 
3. `format` args describes the input format with either of the values ("list", "dict")
4. `component` can be any of the following ('i', 'v', 't', 'iv', 'it','ivt') to compute performance for (instrument, verb, target, instrument-verb, instrument-target, instrument-verb-target) respectively, default is 'ivt' for triplets.<
* the output is a `dict` with keys("AP", "mAP", "Rec", "mRec", "Pre", "mPre") for per-class AP, mean AP, per-class Recall, mean Recall, per-class Precision and mean Precision respectively.


<a name="detectionExample"></a>
#### Example usage

``` python
import ivtmetrics
detect = ivtmetrics.Detection(num_class=100)

network = MyModel(...) # your model here

# training

format = "list"
for epoch in number of epochs:
  for images, labels in dataloader(...): # your data loader
    predictions = network(image)
    labels, predictions = formatYourLabels(labels, predictions)
    detect.update(labels, predictions, format=format)
      
  results_i = detect.compute_AP('i')
  print("instrument per class AP", results_i["AP"])
  print("instrument mean AP", results_i["mAP"])
    
  results_ivt = detect.compute_AP('ivt')
  print("triplet mean AP", results_ivt["mAP"])
  detect.reset()


# evaluation

format = "dict"
for video in videos:
  for images, labels in dataloader(video, ..): # your data loader
    predictions = network(image)
    labels, predictions = formatYourLabels(labels, predictions)
    detect.update(labels, predictions, format=format)
  detect.video_end()
    
results_ivt = detect.compute_video_AP('ivt')
print("triplet mean AP", results_ivt["mAP"])
print("triplet mean recall", results_ivt["mRec"])
print("triplet mean precision", results_ivt["mPre"])
```

Any `nan` value in results is for classes with no occurrence in the data sample.


<br />
<a name="Disentangle"></a>

### Disentangle

Although, the `Detection()` and `Recognition()` classes uses the `Disentangle()` internally, this function can still be used independently for component filtering in the following ways:

``` python
filter = ivtmetrics.Disentangle()
```

Afterwards, each of the component's predictions/labels can be filtered from the main triplet's predictions/labels as follows:

``` python
i_labels = filter.extract(inputs=ivt_labels, component="i")
v_preds  = filter.extract(inputs=ivt_preds, component="v")
t_preds  = filter.extract(inputs=ivt_preds, component="t")
iv_labels = filter.extract(inputs=ivt_labels, component="iv")
it_labels = filter.extract(inputs=ivt_labels, component="it")
```





<a name="Docker"></a>
## Docker

coming soon ..


<a name="Citation"></a>
# Citation

If you use this metrics in your project or research, please consider citing the associated publication:
```
@article{nwoye2022data,
  title={Data Splits and Metrics for Benchmarking Methods on Surgical Action Triplet Datasets},
  author={Nwoye, Chinedu Innocent and Padoy, Nicolas},
  journal={arXiv preprint arXiv:2204.05235},
  year={2022}
}
```




<a name="References"></a>
### References

1. Nwoye, C. I., Yu, T., Gonzalez, C., Seeliger, B., Mascagni, P., Mutter, D., ... & Padoy, N. (2021). Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. arXiv preprint arXiv:2109.03223.
2. Nwoye, C. I., Gonzalez, C., Yu, T., Mascagni, P., Mutter, D., Marescaux, J., & Padoy, N. (2020, October). Recognition of instrument-tissue interactions in endoscopic videos via action triplets. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 364-374). Springer, Cham.
3. http://camma.u-strasbg.fr/datasets
4. https://cholectriplet2022.grand-challenge.org
5. https://cholectriplet2021.grand-challenge.org



## License
```
BSD 2-Clause License

Copyright (c) 2022, Research Group CAMMA
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.```
```

