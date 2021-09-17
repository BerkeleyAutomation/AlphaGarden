## Cicle Tracking & Segmentation Pipeline

### Steps to run and high-level overview

1. Load in uncropped images
2. Verify that all the above folders exist (if not most should be automatically created)
3. Verify that constants in center_constants.py and constants.py. Important constanst include:
    - IMG_DIR = "./inputs" (images to feed into segmentation mode)
    - CIRCLE_PATH = "./circles/" (where to save the CM circles)
    - PRIOR_PATH = "./priors/" (where to find the priors)
    - TEST_MODEL = './models/MOST_RECENT_MODEL.h5' (replace with most recent model)
4. Open ```full_auto_circles.py``` and verify that the main reads:     
```Python    
for f in daily_files("./farmbotsony"):
    process_image("farmbotsony/" + f, True, True) 
```
5. Wait! It's gonna take a while from here, but when you come back everything should be processed. 

### File Structure

``` circles/ ```
- All circles with corresponding types, centers, and radii in cm. Useful for the simulator--stored as pickles. 
- Naming convention: YYMMDD.p

``` farmbotsony/ ```
- Uncropped overhead images
- Naming convention: snc-YYMMDDHHMMSSMS.jpg

``` input/ ```
- Cropped overhead images
- Naming convention: snc-YYMMDDHHMMSSMS.jpg

``` figures/ ```
- Overhead images with circles overlayed
- Naming convention: snc-YYMMDDHHMMSSMS.png

``` model_out/ ``` 
- Various utility images, such as confidence maps, shifted images, and combined images

``` models/ ```
- TensorFlow Segmentation Model

``` models/growth_models ```
- Radius logistic growth models by plant type (stored as Python serialization objects aka pickles)

``` old/ ```
- Old iterations of the pipeline (shouldn't be on Github)

``` other_scripts/ ```
- Deprecated scripts

``` post_process/ ```
- Image Segmentation masks 
- Naming convention: snc-YYMMDDHHMMSSMS.png

``` priors/ ```
- Circles sorted in pixel format used as algorithm priors
- Naming convention: priorsYYMMDD.p

