<h2 align="center">Extracting Dynamic Facial Expressions from Naturalistic Videos</h2>

### Prerequisites 
* CUDA (CPU for facial expression animation)
* [Nest](https://github.com/ZhouYanzhao/Nest.git)
* [Python=3.6.8](https://www.python.org)
* [PyTorch=0.4.1](https://pytorch.org) (make sure to install with cuda and torchvision)
* [Conda](https://www.anaconda.com/)
* [MakeHuman](http://www.makehumancommunity.org/)
* [FACSHuman](https://github.com/montybot/FACSHuman)

### Installation
1. Install all the prerequisities
2. Integrate FACSHuman with MakeHuman(follow the instruction: https://github.com/montybot/FACSHuman)
3. Open MakeHuman and go to Settings/Plugins to enable scripting and shell (for generating facial expression animation)

### Running our method

1. `git clone https://github.com/sccnlab/Extracting-dynamic-facial-expressions-from-naturalistic-videos/ ./ExtractFace`
2. `nest module install ./ExtractFace/ face`
3.  verify the installation `nest module list --filter face`

#### Training

