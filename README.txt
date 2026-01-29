ACORBA v1.3: February 2021

If you used it, give us feedback! Negative feedback is a very important part of development to improve flexibility
and functionality. By the way contact us if you want to be involve (just ideas are part of it too).

Used it and it worked? Cite us: https://www.biorxiv.org/content/10.1101/2021.07.15.452462v1

-------------------------------------------------------------------------

MAJOR FIXES: 

	Error handling in microscope mode has been fixed. If one root bugs from the first timeframe or in the middle 		
	of its analysis, the export of the excel output file is no more crashing.	

	The models for microscopy through method were retrained with more data

	The model architecture for microscopy sandwich has been changed to a custom UNET (keras_unet library)
	and the model was trained with more data than the original

	For miroscope images, another layer of safety for determining the direction of the vector has been 
	added: Now the vector direction is in priority determined by calculating the sum of pixels from vectors
	going from the middle of the skeleton to the left or right of the image. The smallest sum determines where
	is the actual root tip. This avoids some reported and observed inversion of the vector direction during analysis.

-------------------------------------------------------------------------

MINOR FIXES:

	A log file with all the infos from the console will be saved in your experiment folder

	Possibility to modify analysis parameters (scanner: vector size and broken skeleton factor,
	microscope: size of the cropping circle for higher resolution or bigger root image)

	Graphic User Interface reorganization	

	Code annotations, simplifications and improvement

	User Manual update with the new implementations
	
	Added button to link to the project repository "Report a bug"
	
	Added button to open the readme.txt file (this file)
	
	Added button to check for updates and automatically start the download if user agrees to update

	Remove the save angle/time plot function

------------------------------------------------------------------------- 

Work in progress:

	Do you have ideas to make ACORBA better?

	Equivalent of bleach correction in imageJ/FIJI

	Compilation of a Linux version. For now you should work with the python script
	in a dedicated Python/Anaconda environement.

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##BQ88QQB##@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@BDI}|=_`        ``.-:~*v7VyKd&Q#@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@Qa|_                                .:^xy
@@@@@@@@@@@@@@@@@@@#A]:                                          
@@@@@@@@@@@@@@@@gkv-                                             
@@@@@@@@@@@@&1)-               @@@@@@@@@@@@@@@@@@@@@@@@        
@@@@@@@@@Ql,               ,x38@@@@@@@@@@@@@@@@@@@@@@@@#B&dscx^,`
@@@@@@#K:     A          `*#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@#j,     B         sB@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@3"     R       rq#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@v     O      :&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@g`    C    v&@@@@@@@@@@@Um9vdHNfcm9ja3MhX3psb29wMjc=@@@@@@@@@@@@
@#x. A -*tQ@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@#Q8B@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

_______________________________________________________________Older updates________________________________________________________________________________________



ACORBA v1.2: Sept 2021

If you used it, give us feedback! Negative feedback is a very important part of development to improve flexibility
and functionality. By the way contact us if you want to be involve (just ideas are part of it too).

Used it and it worked? Cite us: https://www.biorxiv.org/content/10.1101/2021.07.15.452462v1

-------------------------------------------------------------------------

MAJOR FIXES: 
A few tweaks and smarter coding in the code allowed to increase spead by up to 40% for the scanner angle analysis (segmentation time is similar)

Added an additional traditional method to auto segment roots from scanner

Introduction of a super-accuracy mode for scanner images (smooth tiling and image augmentation)>High RAM and GPU required if ticked 	
	This last method works well with 600dpi pictures but is not recommended (10x longer). 
	Please use grayscale 8bits 1200dpi scanner pictures for better accuracy and faster processing with the regular machine learning method

Update of Tensorflow to 2.6. If you wish to use GPU computation install CUDA 11.2 and copy the file of CUDnn 8.1 in the installation folder

Possibility to either save the segmentations as tif stack files (tick the box in options) or only run the segmentation (button "Segmentation only")
	Be aware that, for microscopy, segmentations are only conducted on 256x256 pictures and thus the segmentations are the same size. For scanner they are
	the original sizes

-------------------------------------------------------------------------

MINOR FIXES:
Code annotations and simplifications

User Manual update with the new implementations

------------------------------------------------------------------------- 

Work in progress:
- Measure of the root pixel lenght (to calculate waving amplitude) and calculations of partial Vertical/Horizontal growth index
- Option for Matching Histogram (bleach correction of a stack)