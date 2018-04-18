# beads-presentation
Presentation for the paper `Audio source separation with magnitude priors: the BEADS model `

The presentation is available [online](https://aliutkus.github.io/beads-presentation/)

## Installation

### Checkout

Be careful to include `reveal.js` in the cloning if you want to try the presentation offline mode. This is done by adding the `--recurse-submodules` in the `git clone` command.

### Interactive jupyter

Hopefully, this should work through conda. Go to the presentation directory, then create the `beads-presentation` environment through:
> conda env create -f environment.yml

then, activate this environment:
* Windows: `activate beads-environment`
* max/linux: `source activate beads-environment`

If that does not work, you will unfortunately have to install the dependencies manually...
In any case, just run the jupyter notebook:

> jupyter notebook

Which will open a browser. The presentation is in the `presentation.ipynb` file. Source code for the model is in the `src/beads.py` file. Note this is not optimized for speed, rather for educational purposes. I should create some optimized version some time.

## Running the presentation

I am using the `RISE` framework. For some reason, it only works well on chromium for me, not firefox that has some cache problems.

There is some issue that is not fixed yet, and that is displaying a warning: `nbagg.transparent is deprecated and ignored. Use figure.facecolor instead.`. You can fix this issue using the trick [https://github.com/matplotlib/jupyter-matplotlib/issues/41#issuecomment-377339581](there). 


## References
If you are using this model or this stuff for your research, please mention the following paper:
> @inproceedings{liutkus2018beads,
>  title={Audio source separation with magnitude priors: the BEADS model},
>  author={Liutkus, Antoine and Rohlfing, Christian and Deleforge, Antoine},
>  booktitle={International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
>  year={2018}
}

The paper is available as a PDF [here](https://hal.inria.fr/hal-01713886/document)
