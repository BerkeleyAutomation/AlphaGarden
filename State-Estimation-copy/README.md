## State Estimation

### Steps to run and high-level overview

1. Load in uncropped images
3. Verify that constants in center_constants.py and constants.py. Important constants include:
    - IMG_DIR = "out/inputs" (images to feed into segmentation mode)
    - CIRCLE_PATH = "out/circles/" (where to save the CM circles)
    - PRIOR_PATH = "out/priors/" (where to find the priors)
    - TEST_MODEL = './models/MOST_RECENT_MODEL.h5' (replace with most recent model)
4. Open ```track.py``` run: ```python3 track.py "snc-<>.jpg"```
5. Wait! It's gonna take a while from here, but when you come back everything should be processed.

### File Structure

``` out/circles/ ```
- All circles with corresponding types, centers, and radii in cm. Useful for the simulator--stored as pickles.
- Naming convention: YYMMDD.p

``` out/overhead/ ```
- Uncropped overhead images
- Naming convention: snc-YYMMDDHHMMSSMS.jpg

``` out/cropped/ ```
- Cropped overhead images
- Naming convention: snc-YYMMDDHHMMSSMS.jpg

``` out/figures/ ```
- Overhead images with circles overlayed
- Naming convention: snc-YYMMDDHHMMSSMS.png

``` out/model_out/ ```
- Various utility images, such as confidence maps, shifted images, and combined images

``` models/ ```
- TensorFlow Segmentation Model

``` models/growth_models ```
- Radius logistic growth models by plant type (stored as Python serialization objects aka pickles)

``` out/post_process/ ```
- Image Segmentation masks
- Naming convention: snc-YYMMDDHHMMSSMS.png

``` out/priors/ ```
- Circles sorted in pixel format used as algorithm priors
- Naming convention: priorsYYMMDD.p
