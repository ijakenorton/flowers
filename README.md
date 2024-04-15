## Classification and Diffusion of flowers

The unet model contains both the auto-encoder and generative diffusion model.
To run the encoder pass `--encoder True` as one of the cmdline args. Run
`python unet.py -h` for help on how to run, or check out `config.py`, this
contains a class Config which is a global configuration object used to keep the
hyperparamters of the models consistent across multiple files, and is used for
both the classifier and unet models. 

These are some examples of configurations
```
  python unet.py --data-aug True --image-size 64 --epochs 100 --image-output-folder flip-horizontal-colours-skip-200 --flip-horizontal True --flip-colour True --noise-range-min 0.0 --noise-range-max 3.0 2> ./errors
  python unet.py --data-aug True --image-size 112 --epochs 200 --image-output-folder flip-horizontal-colours-noise-range-200 --flip-horizontal True --flip-colour True --noise-range-min 0.0 --noise-range-max 3.0 2> ./errors
  python unet.py --data-aug True --image-size 112 --epochs 200 --image-output-folder flip-horizontal-colours-noise-range-less-200 --flip-horizontal True --flip-colour True --noise-range-min 0.0 --noise-range-max 1.0 2> ./errors
  python classifier.py --data-aug True --image-size 256 --learning-rate 0.0001 --fine True --epochs 200 --image-output-folder balanced 2> ./errors
  python classifier.py --data-aug True --image-size 256 --learning-rate 0.0001  --epochs 200 --image-output-folder sparse 2> ./errors
```
I piped out errors to a file to avoid seeing the tensorflow warnings which are
printed for every run.

There is a bug with the argparse where for any of the boolean values, they will
be considered the default (False) if nothing is passed, but True if the cmdline
flag is passed, even if you pass --flip-horizontal False, for example, so for
any parameter you don't want to use, leave it out.

Currently, the images and plots are output to files, as it is easier to track.

There is a utility script, `parse_name.py` which takes a saved model name from
stdin and converts it to the cmdline args used to create it. This has not been
thoroughly tested so may have some issues with some of the files, but is
hopefully helpful in disambiguating the saved models.

Examples
```
ls saved | tail -n 1 | python parse_name.py

echo "IS=64d_FN=0d_EP=200d_NR=(0.0,2.0)s_VB=1d_ENC=0d_RWB=0.1f_RDR=0.4f_RBN=1d_DA=1d_FH=1d_SK=0d_FC=1d_LR=0.01f_IOF=flip-horizontal-colourss_cnn_net.h5" | python parse_name.py
Outputs > 
--image-size=True --fine=True --epochs=True --noise-range-min=0.0 --noise-range-max=2.0 --verbose=True --encoder=True --reg-wdecay-beta=0.1 --reg-dropout-rate=0.4 \
--reg-batch-norm=True --data-aug=True --flip-horizontal=True --skips=True --flip-colour=True --learning-rate=0.01 --image-output-folder=flip-horizontal-colour
```
