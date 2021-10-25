<h2 align="center">Extracting Dynamic Facial Expressions from Naturalistic Videos</h2>

### Prerequisites 
* CUDA (CPU for facial expression animation)
* [Nest](https://github.com/ZhouYanzhao/Nest.git)
* [Python=3.6.8](https://www.python.org)
* [PyTorch=0.4.1](https://pytorch.org) (make sure to install with cuda and torchvision)
* [Conda](https://www.anaconda.com/) (You could use Conda to install all prerequistes)
* [MakeHuman](http://www.makehumancommunity.org/)
* [FACSHuman](https://github.com/montybot/FACSHuman)
* [FFmpeg](https://www.ffmpeg.org/) (You could use conda install)

### Installation
1. Install all the prerequisities
   (Conda Version)
    ```bash
    # set up environment
    $ conda create --name face python=3.6.8
    $ conda activate face
    # In the Nest folder downloaded from https://github.com/ZhouYanzhao/Nest.git
    $ python setup.py install
    # Install Pytorch
    $ conda install pytorch=0.4.1 cuda90 -c pytorch
    $ conda install -c pytorch torchvision
    # Install Progress bar
    $ conda install -c conda-forge tqdm
    # For the later nest command, replace nest with your_directory/anaconda3/envs/face/bin/nest
    
    ```
3. Integrate FACSHuman with MakeHuman(follow the instruction: https://github.com/montybot/FACSHuman)
4. Open MakeHuman and go to Settings/Plugins to enable scripting and shell (for generating facial expression animation)

### Running our method

1. `git clone https://github.com/sccnlab/Extracting-dynamic-facial-expressions-from-naturalistic-videos/ ./ExtractFace`
2. `nest module install ./ExtractFace/ face`
3.  Verify the installation `nest module list --filter face`

#### Data
We use the facial expression data from FEAFA paper (https://www.iiplab.net/feafa/). Please download it from there if you want use the same data

#### Training
1. Change the data_dir in /demo/s3n_train.yml to your data folder
2. Run `CUDA_VISIBLE_DEVICES=0,1 nest task run ./demo/s3n_train.yml`

#### Testing
1. Change the data_dir in /demo/s3n_test.yml to your data folder
2. You can use our pretrained model(https://drive.google.com/file/d/1lzeO-ozXTkm29Kbny_5pekHCPdYg2FgL/view?usp=sharing) and set resume in /demo/s3n_test.yml to the pretrained model
3. Run `CUDA_VISIBLE_DEVICES=0,1 nest task run ./demo/s3n_test.yml`

#### Prediction
1. Create you own facial expression video and run `ffmpeg -i your video name  -r 30 %06d.png` (remove all the blank frames)
2. Change the data_dir in /demo/s3n_pred.yml to your data folder
3. You can use our pretrained model (same instruction above and set resume in /demo/s3n_pred.yml to the pretrained model)
4. Run `CUDA_VISIBLE_DEVICES=0,1 nest task run ./demo/s3n_pred.yml`
5. The final output will save in ./output/pred.npy

### Facial animation generation
1. Change the Base in /facial_animation/makehuman_face.py to your directory you want to save
2. Make sure the directory in np.load is directing to the location of prediction file
3. Copy the code to MakeHuman software /Utilities/Scripting and execute the code in /Utilities/Execute
4. For better eyebrow animation, run /facial_animation/makehuman_face.py (Change the input_directory to your saved direction in makehuman_face.py, set your new custom saved directory, and remove .DS_Store if you have one)
5. To generate the video, run `ffmpeg -framerate 30 -i img%03d.png -c:v libx264 -pix_fmt yuv420p out.mp4` in your final saved directory (you can set custom framerate)

# Questions
For any questions, please raise an issue on GitHub or email wangccy@bc.edu
