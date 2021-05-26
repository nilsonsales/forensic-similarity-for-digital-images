# Source code for: "Forensic Similarity for Digital Images" 
by Owen Mayer and Matthew C. Stamm  
Deparment of Electrical and Computer Engineering  
Drexel University - Philadelphia, PA, USA

Visit the [project webpage](http://omayer.gitlab.io/forensicsimilarity/) for information and interactive demos!

## Status of the project for the DPI subject:
- [x] Present the article and code
- [x] Implement tool to detect image forgery using open-cv


## Prerequisites 
*  python 3
*  python packages:
    *  tensorflow
    *  numpy
    *  tqdm
* optional recommended python packages:
    *  jupyter notebook (for working with example scripts)
    *  matplotlib (for loading images in the examples)
    *  pillow (for loading JPEG images)


**NEW**: The prerequisites are now listed in the 'requirements.txt' file. To install it, run:
```
$ pip install -r requirements.txt
```

## Getting Started

Use the "calculate_forensic_similarity" definition in forensic_similarity.py to calculate the forensic similarity between two image patches.

Please see the [jupyter notebook examples](https://gitlab.com/MISLgit/forensic-similarity-for-digital-images/tree/master/examples) for how to use the forensic similarity code.

## Example of Forgery Detection

<img src="https://github.com/nilsonsales/forensic-similarity-for-digital-images/raw/master/examples/images/detected_2.jpg" width="720">

## Cite this code
If you are using this code for academic research, please cite this paper:

Mayer, Owen, and Matthew C. Stamm. "Forensic Similarity for Digital Images." *IEEE Transactions on Information Forensics and Security* (2019).

bibtex:
```
@article{mayer2019similarity,
  title={Forensic Similarity for Digital Images},
  author={Mayer, Owen and Stamm, Matthew C},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2019},
  publisher={IEEE}
}
```
