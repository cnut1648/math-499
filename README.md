# math-499

team2net



if on google colab

```python
!git clone https://github.com/cnut1648/math-499.git
!pip install git+https://github.com/mjkvaak/ImageDataAugmentor
import sys
sys.path.append('/content/math-499')
import config
import prepare_data
import train

train.train(config.ensemble_num)
# or train for all tranches
train.trainAll()
```

