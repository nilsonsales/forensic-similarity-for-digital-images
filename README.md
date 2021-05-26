# Source code for: "Forensic Similarity for Digital Images" 
by Owen Mayer and Matthew C. Stamm  
Deparment of Electrical and Computer Engineering  
Drexel University - Philadelphia, PA, USA

Visit the [project webpage](http://omayer.gitlab.io/forensicsimilarity/) for information and interactive demos!

## Status of the project for the DPI subject
- [x] Present the article and code
- [x] Implement tool to detect image forgery using open-cv

A tool to visualise forgery detection is implemented in `examples/test_similarity.py`. If you want to use a different image, edit directly in the code.

- After you run the file, select a point on the image and press 'q';
- The selected region will be shown you, press 'q' again to continue;
- Wait for the model to calculate the similarity, it might take a few seconds;
- When it's done, check the result and press 'q' to quit.

### Example of Forgery Detection
<img src="https://github.com/nilsonsales/forensic-similarity-for-digital-images/raw/master/examples/images/original_2.jpg" width="512"> <img src="https://github.com/nilsonsales/forensic-similarity-for-digital-images/raw/master/examples/images/detected_2.jpg" width="512">


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


**NEW**: The prerequisites are now listed in the `requirements.txt` file. To install it, run:
```
$ pip install -r requirements.txt
```

## Getting Started

Use the "calculate_forensic_similarity" definition in forensic_similarity.py to calculate the forensic similarity between two image patches.

Please see the [jupyter notebook examples](https://gitlab.com/MISLgit/forensic-similarity-for-digital-images/tree/master/examples) for how to use the forensic similarity code.


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
