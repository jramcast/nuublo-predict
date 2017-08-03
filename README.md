# nuublo-predict

[![Build Status](https://travis-ci.org/jramcast/nuublo-predict.svg?branch=master)](https://travis-ci.org/jramcast/nuublo-predict)

Weather text classifier based on https://www.kaggle.com/c/crowdflower-weather-twitter/data


## Run
Requires python 3.

```
pip install -r requirements.txt
./server.py
```

## Classifiy text

```
curl -X POST \
 -F text="Partly sunny, breezy with highs in the 80s. Heavy and possibly severe t-storms around and after sunset. Watches could be issued later." \
 http://127.0.0.1:5002/prediction
```

Result (detected classes):
``` 
[
  "WEATHER: sun", 
  "WEATHER: hot", 
  "WEATHER: storms", 
  "WEATHER: wind", 
  "WHEN: future (forecast)", 
  "WHEN: current (same day) weather", 
  "SENTIMENT: neutral / author is just sharing information"
]
```
