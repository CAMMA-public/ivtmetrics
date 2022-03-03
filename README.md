

[![PyPI version](https://badge.fury.io/py/motmetrics.svg)](https://badge.fury.io/py/ivtmetrics)

# ivtmetrics

The **ivtmetrics** library provides a Python implementation of metrics for benchmarking surgical action triplet detection and recognition.

## Features at a glance
- *Recognition Evaluation* <br/>
Provides AP metrics to measure the performance of a model on action triplet recognition. 
- *Detection Evaluation* <br/>
Supports Intersection over Union distances measure of the triplet localization with respect to the instruments.
- *Flexible Analysis* <br/>
  - Supports for switching between frame-wise to video-wise averaging of the AP.
  - Supports disentangle prediction and obtained filtered performance for the various components of the triplets as well as their association performances at various levels. 



<a name="installation"></a>
## Installation
### Install via PyPi
To install **ivtmetrics** use `pip`
```
pip install ivtmetrics
```
Python 3.5-3.9 and numpy and scikit-learn are required.

### Install via Conda
coming soon ...

<a name="Metrics"></a>
## Metrics

The metrics have been aligned with what is reported by [CholecT50](https://arxiv.org/abs/2109.03223) benchmark.
**ivtmetrics** ivtmetrics can be imported in the following way: 

```python
import ivtmetrics

```

The metrics implement both **recognition** and **detection** evaluation.
The metrics internally implement a disentangle function to help filter the triplet components as well as triplet different levels af association.

### Recognition Metrics

**Recognition ivtmetrics** can be used in the following ways:

```python
metric = ivtmetrics.Recognition(num_class)

```
This takes an argument `num_class` which is default to `100`

The following function are possible with the 'Recognition` class:
Name|Description
:---|:---
update(targets, predictions)|takes in a (batch of) vector predictions and their corresponding groundtruth. vector size must match `num_class` in the class initialization.
video_end()|Call to make the end of one video sequence.
reset()|Reset current records. Useful during training and can be called at the begining of each epoch to avoid overlapping epoch performances.
reset_global()|Reset all records. Useful for switching between training/validation/testing or can be called at the begining of new experiment.
compute_AP(component)|Obtain the average precision on the fly. This gives the AP only on examples cases after the last `reset()` call. Useful for epoch performance during training. <ul><li>args `component` can be any of the following ('i', 'v', 't', 'iv', 'it','ivt') to compute performance for (instrument, verb, target, instrument-verb, instrument-target, instrument-verb-target) respectively. default is 'ivt' for triplets.</li> <li>the output is a `dict` with keys("AP", "mAP") for per-class and mean AP respectively.</li></ul>
compute_video_AP(component)|compute video-wise AP performance as used in CholecT50 benchmarks.
compute_global_AP(component)|compute frame-wise AP performance for all seen samples.
topK(k, component) | Obtain top K performance on action triplet recognition for all seen examples. args `k` can be any int between 1-99. k = [5,10,15,20] have been used in benchmark papers.
topClass(k, component)|Obtain top K recognized classes on action triplet recognition for all seen examples. args `k` can be any int between 1-99. k = 10 have been used in benchmark papers.



<a name="recognitionExample"></a>
#### Example usage

```python
metric = ivtmetrics.Recognition(num_class=100)

network = MyModel(...) # your model here

# training
for epoch in number of epochs:
  images, labels = dataloader(...) # your data loader
  predictions = network(image)
  metric.update(labels, predictions)
  
results_i = metrics.compute_AP('i')
print("instrument per class AP", results_i["AP"])
print("instrument mean AP", results_i["mAP"])

results_ivt = metrics.compute_AP('ivt')
print("triplet mean AP", results_ivt["mAP"])


# evaluation
for video in videos:
  for images, labels in dataloader(video, ..): # your data loader
    predictions = network(image)
    metric.update(labels, predictions)
  video_end()
    
results_i = metrics.compute_video_AP('i')
print("instrument per class AP", results_i["AP"])
print("instrument mean AP", results_i["mAP"])

results_it = metrics.compute_video_AP('it')
print("instrument-target mean AP", results_it["mAP"])

results_ivt = metrics.compute_video_AP('ivt')
print("triplet mean AP", results_ivt["mAP"])
```







### Detection Metrics

**Detection ivtmetrics** can be used in the following ways:

```python
metric = ivtmetrics.Detection(num_class, num_tool)

```
This takes an argument `num_class` which is default to `100` and `num_tool` which is default to `6`

The following function are possible with the 'Recognition` class:
Name|Description
:---|:---
update(targets, predictions, format)|input: takes in a (batch of) list/dict predictions and their corresponding groundtruth. Each frame prediction/groundtruth can be either as a `list of list` or `list of dict`. <ul><li>(a) **list of list format**: [[tripletID, toolID, toolProbs, x, y, w, h], [tripletID, toolID, toolProbs, x, y, w, h], ...] where: <ul><li>`tripletID` = triplet unique identity</li><li>`toolID` = instrument unique identity</li><li>`toolProbs` = instrument detection confidence</li><li>`x` = bounding box x1 coordiante</li><li>`y` = bounding box y1 coordinate</li><li>`w` = width of the box</li><li>`h` = height of the box</li><li>The [x,y,w,h] are scaled between 0..1 </li></ul>.<li>(b) **list of dict format**: [{"triplet":tripletID, "instrument":[toolID, toolProbs, x, y, w, h]}, {"triplet":tripletID, "instrument":[toolID, toolProbs, x, y, w, h]}, ...]. </li><li>The `format` args describes the input format with either of the values ("list", "dict")</li></ui>
video_end()|Call to make the end of one video sequence.
reset()|Reset current records. Useful during training and can be called at the begining of each epoch to avoid overlapping epoch performances.
reset_global()|Reset all records. Useful for switching between training/validation/testing or can be called at the begining of new experiment.
compute_AP(component)|Obtain the average precision on the fly. This gives the AP only on examples cases after the last `reset()` call. Useful for epoch performance during training.<ul><li>args `component` can be any of the following ('i', 'v', 't', 'iv', 'it','ivt') to compute performance for (instrument, verb, target, instrument-verb, instrument-target, instrument-verb-target) respectively.</li> <li>default is 'ivt' for triplets.</li><li> the output is a `dict` with keys("AP", "mAP", "Rec", "mRec", "Pre", "mPre") for per-class AP, mean AP, per-class Recall, mean Recall, per-class Precision and mean Precision respectively.</li></ui>
compute_video_AP(component)|compute video-wise AP performance as used in CholecT50 benchmarks.
compute_global_AP(component)|compute frame-wise AP performance for all seen samples.




<a name="detectionExample"></a>
#### Example usage

```python
metric = ivtmetrics.Detection(num_class=100)

network = MyModel(...) # your model here

# training
format = "list"
for epoch in number of epochs:
  images, labels = dataloader(...) # your data loader
  predictions = network(image)
  labels, predictions = formatYourLabels(labels, predictions)
  metric.update(labels, predictions, format=format)
  
results_i = metrics.compute_AP('i')
print("instrument per class AP", results_i["AP"])
print("instrument mean AP", results_i["mAP"])

results_ivt = metrics.compute_AP('ivt')
print("triplet mean AP", results_ivt["mAP"])


# evaluation
format = "dict"
for video in videos:
  for images, labels in dataloader(video, ..): # your data loader
    predictions = network(image)
    labels, predictions = formatYourLabels(labels, predictions)
    metric.update(labels, predictions, format=format)
  video_end()
    
results_ivt = metrics.compute_video_AP('ivt')
print("triplet mean AP", results_ivt["mAP"])
print("triplet mean recall", results_ivt["mRec"])
print("triplet mean precision", results_ivt["mPre"])
```


<a name="Docker"></a>
## Docker
coming soon ..




<a name="References"></a>
### References
1. Nwoye, C. I., Yu, T., Gonzalez, C., Seeliger, B., Mascagni, P., Mutter, D., ... & Padoy, N. (2021). Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. arXiv preprint arXiv:2109.03223.



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

