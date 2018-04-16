# beads-presentation
Presentation for the paper `Audio source separation with magnitude priors: the BEADS model `

The presentation is available [online](https://aliutkus.github.io/beads-presentation/)

## References
If you are using this model or this stuff for your research, please mention the following paper:
> @inproceedings{liutkus2018beads,
>  title={Audio source separation with magnitude priors: the BEADS model},
>  author={Liutkus, Antoine and Rohlfing, Christian and Deleforge, Antoine},
>  booktitle={International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
>  year={2018}
}

The paper is available as a PDF [here](https://hal.inria.fr/hal-01713886/document)

## Installation for the source code
Through conda:
> conda env create -f environment.yml


Then, you can run the jupyter console:

> jupyter console

Which will open a browser. The presentation is in the `presentation.ipynb` file. Source code for the model is in the `src/beads.py` file. Note this is not optimized for speed, rather for educational purposes. I should create some optimized version some time.

## Running the presentation

I am using the `RISE` framework. For some reason, it only works well on chromium for me, not firefox that has some cache problems.
