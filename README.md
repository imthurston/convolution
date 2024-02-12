# convolution.py
Incorporate convolution methods and classes for determining phase boundaries of chemical systems.
This package is intended for use within a jupyter notebook for display and retrieval of associated images and graphs. 

# Classes:
## ***UniverseConvolution***: An object useful for averaging multiple frames of an MDAanalysis Universe object.
### Attributes:
1. UniverseConvolution.universe
    - *MDAnalysis universe object*
2. UniverseConvolution.size
    - *Number of frames in the universe object*
### Methods:
```python
def create_ave_df(self, travel_axis="Z", molecule="UNK", bins=100, method="COM", frame_start=0, frame_end=False)
```
* *Creates a pandas dataframe containing the concentration data, centered on the phase boundary, averaged across all the specified frames.*
```python
def plot_ave_conc_scatter(self, travel_axis="Z", molecule="UNK", bins=100, method="COM", frame_start=0, frame_end=False)
```
* *Plots a scatter plot from the data created from* `create_ave_df()`. 
### General class arguments:
* ***travel_axis***: The axis along which you want to create bins
* ***molecule***: The molecule for which you're trying to analyze.
* ***bins***: The number of bins used to partition the cell for integration. When using `method="Area"`, ~1000 bins is appropriate. When using `method="COM"` (center of mass), ~100 bins is appropriate. This depends on cell size, make sure not to use too many or to few bins.
* ***method*** Select the method you want to use for concentration/fraction analyis. `"COM"` will calculate the center of mass for each molecule, and place it into each bin from the center of mass. `"Area"` will calculate the area of a species at a given slice, expand it by the bin width into a volume, then take the fraction of that species of total bin volume.
* ***frame_start*** The frame in which you want to start for the average.
* ***frame_end*** The frame in which you want to end for the average. If `frame_end == False ` then the end frame will be the last frame in the trajectory file.

## ***FrameConvolution***: Object for centering the phase boundary of a frame.
### Atributes
1. FrameConvolution.universe
    - *MDAnalysis universe object.*
2. FrameConvolution.frame_number
    - *The trajectory frame number from which the system was created.*
3. FrameConvolution.df
     - *A pandas dataframe containing information about the atoms in the system. Similar in format to a .gro file, with some additional information.*
     - ***When create a FrameConvolution object, this data gets saved to a .csv file. You may need to clear this data if you're not running analyses concurrently, or you could create errors.***
4. FrameConvolution.cell_conc_df
    - *A dataframe object containing the concentrations of various species within the mixture*
### Additional class arguments 
##### *(in addition to `UniverseConvolution`)*
* ***