# HyperRGBD

[ ![License] [license-image] ] [license]

[license-image]: https://img.shields.io/badge/license-gpl-green.svg?style=flat
[license]: https://github.com/aliosciapetrelli/Pairwise3DRegistrationEvaluation/blob/master/LICENSE

Description
-----------
C++ framework for building new datasets by aggregating images from different existing RGB-D datasets. The framework is described in:

Petrelli A., Di Stefano L., "Learning to Weight Color And Depth for RGB-D Image Search", Submitted at International Conference on Image Analysis and Processing, 2017.

[//]: # " [Petrelli A., Di Stefano L., "Learning to Weight Color And Depth for RGB-D Image Search", European Conference on Computer Vision, 2016.](http://onlinelibrary.wiley.com/doi/10.1111/cgf.12732/epdf)"

Webpage
-----------
http://www.vision.deis.unibo.it/research/78-cvlab/107-hyperrgbd

Usage
-----------
The integration of an existing dataset into the framework is accomplished by implementing the *IDataset* interface that requires the implementation of methods:
* *ReadImage()* loads from disk an image in the required format (e.g. RGB, depth map, 3D point cloud, mask image etc.).
* *GetTrainingSet()* returns the training set as a list of images, each one denoted by the associated filename and label.
* *GetTestSet()* returns the test set as a list of images, each one denoted by the associated filename and label.

The aggregation of datasets into new ones is enabled by the *HyperDataset* class, that requires:
* The definition of the mapping between the categories of the existing datasets and those of the aggregated one through the *m_mapCatAssociations* map.
* Each aggregated dataset implements a version of the *GetTrainingSet()* method that returns all the images comprising the dataset.
* The definition of a criteria for splitting the training and test set through the *GetTrainingSet()* and *GetTestSet()* methods.

A few examples can be found in [hyperrgbdTestMain.cpp](https://github.com/aliosciapetrelli/HyperRGBD/blob/master/hyperrgbdTestMain.cpp)

Dependencies
-----------
The framework requires [OpenCV](http://opencv.org/) library for handling images. Moreover, depth maps and calibration data of [BigBIRD](http://rll.berkeley.edu/bigbird/) dataset are stored in [HDF5](https://www.hdfgroup.org/HDF5/) format.

The code has been tested on Windows 7 and Microsoft Visual Studio 2010.
