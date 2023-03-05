# NanoViT

NanoViT is an implementation from scratch of the Vision Transformer architecture using PyTorch. It is based on the implementation of the Time library and downloads pre-trained weights of the Vision Transformer from that library. This implementation is designed for vision tasks such as image classification, object detection, and semantic segmentation.

## Installation

To run NanoViT, you need to create a Conda environment using the environment.yml file at the root of the project. You can do this with the following command:

``` bash
conda env create -f environment.yml
```
This will create a new environment called NanoViT.

### Usage

To use NanoViT, you can run the forward.py script inside the src/ directory. This script loads a pre-trained Vision Transformer model and performs forward inference on an image.

To run the script, activate the NanoViT environment:

```bash
conda activate NanoViT
```
Then, run the script with the following command:

```python
python src/forward.py
```

The script will load the pre-trained model, preprocess the input image, and perform forward inference. The output will be the predicted class label for the input image.

### Credits

NanoViT is based on the implementation of the Vision Transformer architecture in the Time library. We acknowledge and thank the developers of the Time library for their work.
Lastly, good portion of the code has been taken from here: https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/vision_transformer 

License

NanoViT is released under the MIT license. See the LICENSE file for more details.