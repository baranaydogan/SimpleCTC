/*
--------------Computes the Euclidean distance transform--------------

Distance transform is based on Meijster's algorithm
The algorithm is explained in:
http://fab.cba.mit.edu/classes/S62.12/docs/Meijster_distance.pdf
Variable names are the same as the article.
_____________________________________________________________

Signed functions return negative distances for background
and positive for foreground. If signed are not used only positive
distances for foreground are returned.
_____________________________________________________________

OUTPUT:
				dt: 	Euclidean distance transform
INPUTS:
				b:  	Binary input image
				m:		number of rows
				n:		number of columns
				p: 		number of slice
_____________________________________________________________
 */

#ifndef DISTANCETRANSFORM_H_
#define DISTANCETRANSFORM_H_

#include <stdint.h>
#include <math.h>
#include <vector>

void DistanceTransform(double* dt, uint8_t* b, int32_t m, int32_t n);

void DistanceTransform(double* dtz, uint8_t* b, int32_t m, int32_t n, int32_t p);

void DistanceTransform_signed(double* dt, int8_t* b, int32_t m, int32_t n);

void DistanceTransform_signed(double* dtz, int8_t* b, int32_t m, int32_t n, int32_t p);

#endif
