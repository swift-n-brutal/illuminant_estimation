[gs568]: http://www.cs.sfu.ca/~colour/data/shi_gehler/

# Data Preparation

## Gehler-Shi Dataset

1. Download and unzip all PNG images, measured illumination files and coordinates files from [Gehler-Shi][gs568].
2. Run the following command to mask out color-checker board and convert 12-bit PNG files to float32 binary files.
    ```
    matlab -nojvm -nodisplay -nosplash -r 'process_gs568'
    ```
Please prepare your raw data as follows and change `DATA_DIR` in the script accordingly:
```
--- DATA_DIR
     |- png
     |   |- xxx.png
     |- groundtruth_568
     |   |- real_illum_568..mat
     |- coordinates
         |- xxx_macbeth.txt
```
    NOTE: Each binary file begins with three int32 values representing height (h), width (w) and number of channels (c) respectively, and is followed by h\*w\*c float32 values in the order of matlab.

3. Randomly split the images into three subsets for cross validation. Here we provide an example: [set\_0](gs568/set_0.txt), [set\_1](gs568/set_1.txt) and [set\_2](gs568/set_2.txt).
4. Run the following command to get the location of patches without zero values in each image. We need to get rid of zero values since they will cause numeric errors when converting pixel values from RGB to UV format.
    ```
    cd .. && python get_locs_gs568.py && cd data
    ```
5. Finally, the binary image files will be stored in `gs568` and the location files will be stored in `gs568/loc`.
