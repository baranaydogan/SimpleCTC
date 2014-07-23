#include "SimpleCT.h"

class CTC {

public:

	CTC(ContourTree* inputCT);
	double getCTC();
	~CTC();

private:

	ContourTree* 	CT; 			// Contour tree

};
