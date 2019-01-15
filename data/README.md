[gs568]: http://www.cs.sfu.ca/~colour/data/shi_gehler/

# Data Preparation

## Gehler-Shi Dataset

1. Download and unzip all PNG images and measured illumination files from [Gehler-Shi][gs568].
2. Run the following command to mask out color-checker board and convert 12-bit PNG files to float32 binary files. Please change the data path in the script according to your situation.
    ```
    matlab -nojvm -nodisplay -nosplash -r 'process_gs568'
    ```
    NOTE: Each binary file begins with three int32 values representing height (h), width (w) and number of channels (c) respectively, and is followed by h\*w\*c float32 values in the order of matlab.

3. Randomly split the images into three subsets for cross validation. Here we provide an example: [set\_0](gs568/set_0.txt), [set\_1](gs568/set_1.txt) and [set\_2](gs568/set_2.txt).
4. Run the following command to get the location of patches without zero values in each image. We need to get rid of zero values since they will cause numeric errors when converting pixel values from RGB to UV format.
    ```
    cd .. && python get_locs_gs568.py && cd data
    ```
