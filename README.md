# Face Completion using GAN

This project aims to complete incomplete faces using Generative Adversarial Networks (GANs). GANs are a class of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a zero-sum game framework. One network generates candidates and the other evaluates them.

## Overview

This project uses a pre-trained GAN model to complete incomplete faces. The incomplete faces are input to the GAN, which then generates complete faces based on its learned patterns from the training data. The completed faces are saved as output images.

## Installation

To run this project, you need to have Python installed on your system. You also need to install the following dependencies:

- torch
- torchvision
- numpy
- Pillow

You can install these dependencies using pip:
```
pip install torch torchvision numpy Pillow
```
## Usage

1. Clone this repository to your local machine:
```
git clone https://github.com/mehrshadina/face-completion-gan.git
```

2. Navigate to the project directory:
```
cd face-completion-gan
```

3. Place your incomplete face images in the `input` directory.

4. Run the completion script:
```
python complete_faces.py
```

5. The completed faces will be saved in the `output` directory.

## Credits

This project uses the SPADE model from NVLabs for face completion. You can find more information about SPADE [here](https://github.com/NVlabs/SPADE).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
