# DDSRNet

![DDSRNet Architecture](./model/DDSRNet.png)

## Overview

DDSRNet (A Deep Model for Denoising and Super-Resolution) is a new two-stage learning architecture designed for simultaneous image denoising and super-resolution. By decoupling these complex tasks, DDSRNet enables specialized processing for each, leading to more effective noise removal and higher-quality upscaling
### Features


## Paper

This implementation is based on our paper published at the **2025 Seventeenth International Conference on Quality Control by Artificial Vision; 1373705 (2025)**: [Read the Paper on arXiv](https://doi.org/10.48550/arXiv.2509.01332)

## Citation
```bibtex
@inproceedings{messai2025enhancing,
  title={Enhancing image quality and anomaly detection for small and dense industrial objects in nuclear recycling},
  author={Messai, Oussama and Zein-Eddine, Abbass and Bentamou, Abdelouahid and Picq, Micka{\"e}l and Duquesne, Nicolas and Puydarrieux, St{\'e}phane and Gavet, Yann},
  booktitle={Seventeenth International Conference on Quality Control by Artificial Vision},
  volume={13737},
  pages={21--28},
  year={2025},
  organization={SPIE}
}
```

## Requirements

Ensure you have the following dependencies installed:

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/)
- [TensorBoardX](https://github.com/lanpa/tensorboardX)
- [torchsummary](https://github.com/sksq96/pytorch-summary)
- SciPy
- NumPy
- Pillow
- Matplotlib
- YAML

You can install all the required packages using the provided `requirements.txt` file.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

