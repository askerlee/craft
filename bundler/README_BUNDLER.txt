BUNDLER APPLICATION FOR MPI-SINTEL
==================================

Copyright (c) 2012 Jonas Wulff, Max-Planck Institute for Intelligent Systems

For questions and comments contact sintel@tue.mpg.de.



INTRODUCTION
============

This readme file describes the usage of the Bundler application, used to 
generate a single datafile from computed optical flow maps. This file can be
used for submission to the evaluation website.

The bundling process consists of three steps:
1)  The canonical frames (one per sequence) are added to the file, to generate
    the visual results on the website
2)  A fixed, randomized subsample of the data is extracted to reduce the amount
    of data to transmit to the evaluation website.
3)  The data is compressed using LZMA (see www.7-zip.org for details).

The main Bundler directory contains self-contained binaries for Linux-x64
(compiled using Ubuntu 12.04, 64-bit), OSX (compiled using OSX 10.8), and
Windows (compiled using CYGWIN under Windows 7, 64 bit).

The Bundler requires two directories for the results on the Clean and Final
pass, respectively. Each of these directories should contain one directory
for each of the sequences:
	ambush_1
	ambush_3
	bamboo_3
	cave_3
	market_1
	PERTURBED_market_3
	market_4
	mountain_2
	PERTURBED_shaman_1
	temple_1
	tiger
	wall

Each of those directories should contain the optical flow fields as .flo files
(see README.txt in the data archives).

Example:
clean/ambush_1/frame0001.flo
              /frame0002.flo
              ...
     /ambush_3/frame0001.flo
              /frame0002.flo
              ...
     ...
final/ambush_1/frame0001.flo
              /frame0002.flo
              ...
     /ambush_3/frame0001.flo
              /frame0002.flo
              ...
     ...



USAGE
=====

Call bundler as:

	bundler DIR_CLEAN DIR_FINAL OUTFILE

with
	DIR_CLEAN   --  Directory containing the optical flow for
                    all clean pass sequences in the test set.
	DIR_FINAL	--  Directory containing the optical flow for
			        all clean pass sequences in the test set.
	OUTFILE		--  Filename to store the extracted results in.
			        Submit this file.
			        
Example:
    
    bundler /home/user/flow/clean/ /home/user/flow/final/ ./out.lzma
    

Note for windows users:
    Since the windows version was compiled using CYGWIN, the paths should be
    given using forward slashes ("/") instead of backslashes ("\").



FURTHER INFORMATION
===================

More information and the data itself can be obtained from 
    http://sintel.is.tue.mpg.de.

The dataset is published as
    Butler, D., Wulff, J., Stanley, G., Black, M.:
    "A naturalistic open source movie for optical flow evaluation", ECCV 2012
    
A more technical account can be found in
    Wulff, J., Butler, D., Stanley, G., Black, M.:
    "Lessons and insights from creating a synthetic optical flow benchmark",
    ECCV 2012, Workshop on Unsolved Problems in Optical Flow and Stereo 
    Estimation
    
If you use this work, please cite:

@inproceedings{Butler:ECCV:2012,
  title = {A naturalistic open source movie for optical flow evaluation},
  author = {Butler, D. J. and Wulff, J. and Stanley, G. B. and Black, M. J.},
  booktitle = {European Conf. on Computer Vision (ECCV)},
  editor = {{A. Fitzgibbon et al. (Eds.)}},
  publisher = {Springer-Verlag},
  series = {Part IV, LNCS 7577},
  month = {oct},
  pages = {611--625},
  year = {2012}
}

