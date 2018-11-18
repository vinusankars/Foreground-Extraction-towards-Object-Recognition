READ ME
=======

SLIC implementation.

Run python slic.py on terminal.
Output files will be created in same folder.

Requirements
============
Python 3
Numpy
OpenCV
Skimage

slic.py implements the SLIC [1] - Vinu Sankar S.

The results for an image is in folder human.
Input image is 2007_009654.jpg to slic.py
It can be changed by varying the img variable in slic.py and adding the image to current directory.
human folder has patches and segments folders containing results per each iteration.
human.jpg is the resized input image.
module_slic.png is the output image from SLIC module inbuilt in Python library skimage.

Directory tree
==============

slic_implementation
|
|--2007_009654.jpg
|--slic.py (executable)
|--README.txt
|--human
--------|
--------|--human.png
--------|--module_slic.png
--------|--patches
----------------|
----------------|--Itr_1.png to Itr_10.png
--------|--segments
----------------|
----------------|--Itr_1.png to Itr_10.png

Comaprisons
===========

The implementation takes 30 seconds per iteration for getting output on average. (I am performing 10 iterations, total 5 minutes)
The SLIC Skimage module takes only 5 seconds maximum for the same.

Quality of patches obtained by the implementation is better than that of the actual module as the edges are preserved very well in my implementation.
But there are some noises in my implementation that can be filtered out, again increasing computations.

Reference
=========

1. R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua, and S. Susstrunk. SLIC
   Superpixels. Technical report, EPFL, 2010.
2. https://github.com/laixintao/slic-python-implementation/blob/master/
