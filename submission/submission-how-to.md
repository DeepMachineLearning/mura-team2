https://worksheets.codalab.org/worksheets/0x42dda565716a4ee08d61f0a23656d8c0/

However, it doesn't tell you how to set up the `src` folder that contains your scripts and models.
Use https://github.com/codalab/codalab-worksheets/wiki/CLI-Basics to set up codalab cli in your local system.

after setting up, use `cl upload src` to upload your scripts and model files as a bundle

Note that codalab cli only supports python 2.7

Then you can use `cl w {your-worksheet}` to switch to the worksheet you are using.

Then execute the following scripts.
Note: 
    - assuming you are in the parent folder of folder `src`
    - replace `{docker image name}` with the name of the docker build from [Docker Hub](https://hub.docker.com/) that you want to use. I built one with Python 3.6, Keras 2.2.0, NVIDIA CUDA/Cudnn, and the packages I needed. Feel free to use it or copy and modify it to your own version: [madcarrot/docker-keras-full](https://hub.docker.com/r/madcarrot/docker-keras-full/)
    - replace `{your model name}` with the actual name of your model):
```
cl add bundle mura-utils//valid .
cl add bundle mura-utils//valid_image_paths.csv .
cl upload src
cl run valid_image_paths.csv:valid_image_paths.csv MURA-v1.1:valid src:src "python src/predict.py valid_image_paths.csv predictions.csv" -n run-predictions --request-docker-image {docker image name} --request-gpus 1
cl make run-predictions/predictions.csv -n predictions-{your model name}
cl macro mura-utils/valid-eval-v1.1 predictions-{your model name}
cl edit predictions-{your model name} --tags mura-submit
```

Lastly, email Mr. Irvin at jirvin16@stanford.edu with a link to your `predictions-{your model name}`. You should describe your model as 
```{Name_of_model} (single model / ensemble) (Institution) optional_http_link```