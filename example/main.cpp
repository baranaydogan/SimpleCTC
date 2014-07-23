#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <string.h>
#include <utility>
#include <algorithm>
#include <omp.h>

#include <time.h>
#include <sys/time.h>

#include "BoundaryDetection.h"
#include "DistanceTransform.h"
#include "SimpleCT.h"
#include "SimpleCTC.h"

using namespace std;

int main( int argc, const char** argv ) {

	if (argc != 4) {
		std::cout << "./simpleExample test_128x128.binary 0.1 32" << std::endl;
		return false;
	}

	struct timespec t1, t2;

	clock_gettime(CLOCK_MONOTONIC,  &t1);


	unsigned short dim = 2;

	char fileName[512];
	strcpy(fileName,argv[1]);

	char *xSize, *ySize, *zSize, *type;

	std::cout << std::endl;

	std::cout << "Input file: \t \t \t " << fileName << std::endl;

	std::cout << "Image dimensions: \t " ;

	type  	= strstr(fileName,".");
	*type 	= '\0';
	type++;

	xSize 	= strstr(fileName,"_");
	*xSize 	= '\0';
	xSize++;

	ySize 	= strstr(xSize,"x");
	*ySize 	= '\0';
	ySize++;

	zSize 	= strstr(ySize,"x");
	if (zSize != NULL) {
		*zSize = '\0';
		zSize++;
		dim = 3;
	}

	std::vector<unsigned int> size;
	size.push_back(atoi(xSize));
	size.push_back(atoi(ySize));

	std::cout << "\t x = " << size[0];
	std::cout << "\t y = " << size[1];

	int numberOfElements 	= size[0]*size[1];

	if (dim == 3) {
		size.push_back(atoi(zSize));
		numberOfElements 	= numberOfElements*size[2];
		std::cout << "\t z = " << size[2];
	}

	std::cout << std::endl;

	std::cout << "File type: \t \t \t " << type << std::endl;

	double* img;
	img 	= new double[numberOfElements];

	FILE * inputImageFile;
	int numberOfReadBytes;
	inputImageFile 	= fopen (argv[1],"r");

	if ( strcmp(type,"uint8") == 0 ) {


		unsigned char* uint8_t_type;
		uint8_t_type 			= new unsigned char[numberOfElements];

		numberOfReadBytes  		= fread (uint8_t_type, 1, numberOfElements, inputImageFile);
		if (numberOfReadBytes != numberOfElements) {
			fputs ("Reading error",stderr);
			exit (1);
		}

		for (int i = 0; i < numberOfElements; ++i)
			img[i] = double(uint8_t_type[i]);

		delete[] uint8_t_type;

	} else if ( strcmp(type,"uint16") == 0 ) {

		unsigned short* uint16_t_type;
		uint16_t_type 			= new unsigned short[numberOfElements];
		numberOfReadBytes  		= fread (uint16_t_type, 2, numberOfElements, inputImageFile);
		if (numberOfReadBytes != numberOfElements) {
			fputs ("Reading error",stderr);
			exit (1);
		}

		for (int i = 0; i < numberOfElements; ++i)
			img[i] = double(uint16_t_type[i]);

		delete[] uint16_t_type;

	} else if ( strcmp(type,"binary") == 0 ) {

		unsigned char* uint8_t_type;
		uint8_t_type 			= new unsigned char[numberOfElements];
		numberOfReadBytes  		= fread (uint8_t_type, 1, numberOfElements, inputImageFile);
		if (numberOfReadBytes != numberOfElements) {
			fputs ("Reading error",stderr);
			exit (1);
		}

		signed char* regionImg;
		regionImg 				= new signed char[numberOfElements];

		if (dim == 2)
			MarkRegions(regionImg, uint8_t_type, size[0], size[1]);
		else
			MarkRegions(regionImg, uint8_t_type, size[0], size[1], size[2]);

		delete[] uint8_t_type;

		double* edtImg;
		edtImg = new double[numberOfElements];

		if (dim == 2)
			DistanceTransform_signed(edtImg, regionImg, size[0], size[1]);
		else
			DistanceTransform_signed(edtImg, regionImg, size[0], size[1], size[2]);

		for (int i = 0; i < numberOfElements; ++i)
			img[i] = double(edtImg[i]);

		delete[] regionImg;
		delete[] edtImg;

	} else {

		std::cerr << "Unknown file type : " << type << std::endl;
		return false;
	}
	fclose (inputImageFile);

	ContourTree ct(img,size);
	ct.prune(uint(atoi(argv[3])), double(atof(argv[2])));

	std::cout << std::endl;
	ct.print_Short();
	std::cout << std::endl;

	CTC ctc(&ct);

	cout<<"Intensity threshold: \t\t "<< double(atof(argv[2])) << endl;
	cout<<"Area/volume threshold: \t\t "<< uint(atoi(argv[3])) << endl << endl;

	cout<<"CTC:\t\t\t\t "<< ctc.getCTC() << endl << endl;

	delete[] img;

	clock_gettime(CLOCK_MONOTONIC,  &t2);

	cout<<"Total computation time:\t\t "<< (t2.tv_sec - t1.tv_sec) + (double) (t2.tv_nsec - t1.tv_nsec) * 1e-9 << endl;
	std::cout << std::endl;

	return true;

}
