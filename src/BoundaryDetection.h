/*
------Marks regions for a given binary input image as follows-----

Foreground 	=  1
Boundary 		=  0
Background	= -1

Uses 4-connectivity for 2D, 6-connectivity for 3D
_____________________________________________________________

OUTPUT:
				dt: 	Region labels
INPUTS:
				b:  	Binary input image
				m:		number of rows
				n:		number of columns
				p: 		number of slices
_____________________________________________________________
 */

#ifndef BOUNDARYDETECTION_H_
#define BOUNDARYDETECTION_H_

#include <stdint.h>

void MarkRegions(int8_t* dt, uint8_t* b, int32_t m, int32_t n);

void MarkRegions(int8_t* out, uint8_t* inp, int32_t m, int32_t n, int32_t p);


#endif
