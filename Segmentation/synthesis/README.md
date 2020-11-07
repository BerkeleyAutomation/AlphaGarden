# Garden-Bed-Synthesization

This repo contains a synthesizier that can generate garden bed images for the leaf segmentation task in the AlphaGarden research project. It takes in a `config.json` file containing the following specifications:
- `background`: path to the background image file
- `background_mask`: path to the ground truth mask of the background
- `leaves`: a nested list of 2-element lists containing (path to single occulated leaf image, oridinal leaf type)
- `encodings`: a correspondence between the ordinal leaf types and their mask colors
- `iterations`: number of additional leaves we would like to include in the synthesization process
- `num_copies`: number of images we would like to synthesize
- `dim`: the side length of the square ROI we would like to extract from the background as the frame for synthesis

During the synthesis process, each leaf is applied the following set of augmentations:
- A uniformly random location across the dimensions of the background
- A uniformly random degrees of rotation
- A uniformly random resizing between 0.75x and 1.25x

The resulting synthesized patches, along with their masks, and the original patch/mask from which they come from will all be found within the folder `generated` at the root directory of this project after calling the script.

**To run the script (given all the prerequisites are satisfied)**, at the project root level, simply call `python3 synthesize.py`.

You can find an example below using 1 overhead image as the original background (with mask), and 8 occulated leaves for 3 different types (nasturtium: 0, borage: 1, bokchoy: 2) (5 additional leaves per image for a total of 10 synthesized images):

The file `sample_config.json` contains the configurations used in generating one of the patches below. 

Here's the **original overhead** image and its mask:

<img src="./demo_images/original.png" width="48%" height="48%">

<img src="./demo_images/mask.png" width="48%" height="48%"/>

Below you can find a randomly 512x512 synthesized background with the leaves overlayed on top:

Here's the **original** patch and its mask: 

<div style="display: flex;">
  <img src="./demo_images/original_patch.png" width="25%" height="25%" /> 
  <img src="./demo_images/original_patch_mask.png" width="25%" height="25%" />
</div>

Here's the **synthesized** patch and its mask:

<div style="display: flex">
  <img src="./demo_images/synthesized_patch.png" width="25%" height="25%" />
  <img src="./demo_images/synthesized_patch_mask.png" width="25%" height="25%" />
</div>
