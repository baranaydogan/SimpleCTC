#include "SimpleCTC.h"
#include <climits>
#include <cmath>
#include <slepceps.h>
#include <iostream>
#include <fstream>

#define PI 3.141592653589793238462643383279502884L

CTC::CTC(ContourTree* inputCT) {
	CT 	= inputCT;
}

CTC::~CTC() {

}

double CTC::getCTC() {

	if (CT->getCT()->size() == 2)
		return 1;

	std::vector<int> fromVertex(CT->getCT()->size(),0);
	std::vector<int> toVertex(CT->getCT()->size(),0);

	std::vector<CT_Vertex*>::iterator CT_it;

	double* img = CT->getImg();

	int newVertex = -1;
	uint replaceVertex;

	for(uint i=0; i < CT->getCT()->size(); i++){

		fromVertex[i] 	= ++newVertex;
		replaceVertex = CT->getCT()->at(i)->index;

		for(uint j=0; j < CT->getCT()->size(); j++)
			if (CT->getCT()->at(j)->toVertex->index == replaceVertex)
				toVertex[j] 		= newVertex;

	}

	for(uint i=0; i < CT->getCT()->size(); i++){

		int supplementAmount = std::abs(round( img[CT->getCT()->at(i)->index] ) - round( img[CT->getCT()->at(i)->toVertex->index] )) - 1;

		if (supplementAmount > 0) {

			toVertex.push_back(toVertex[i]);
			toVertex[i] 		= newVertex + 1;

			newVertex   	= newVertex + supplementAmount;
			fromVertex.push_back(newVertex);

			for (int n=1; n<supplementAmount; n++) {
				fromVertex.push_back(newVertex-n);
				toVertex.push_back(newVertex-n+1);
			}

		}

	}

	int N 			= fromVertex.size();


	SlepcInitialize(NULL,NULL,NULL,NULL);

	Mat            	Laplacian;
	MatCreate(PETSC_COMM_WORLD,&Laplacian);
	MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, 0, PETSC_NULL, 0, PETSC_NULL, &Laplacian);

	PetscScalar tmp;

	for (int i=0; i<N; i++) {

		tmp = -1;
		MatSetValues(Laplacian, 1, &fromVertex[i], 	1, &toVertex[i],  &tmp, 	INSERT_VALUES);
		MatSetValues(Laplacian, 1, &toVertex[i], 	1, &fromVertex[i],  &tmp, 	INSERT_VALUES);

		tmp = 0;
		for (int j = 0; j<N; j++)
			if ( (i != j) && (toVertex[j] == fromVertex[i]) )
				tmp++;
			else if ( (i == j) && (toVertex[j] == fromVertex[i]) )
				tmp--;

		tmp++;

		MatSetValues(Laplacian, 1, &fromVertex[i], 	1, &fromVertex[i],  &tmp, 	INSERT_VALUES);

	}

	MatAssemblyBegin(Laplacian,	MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(Laplacian,		MAT_FINAL_ASSEMBLY);

	EPS eps;
	EPSCreate(PETSC_COMM_WORLD,&eps);
	EPSSetOperators(eps,Laplacian,NULL);
	EPSSetProblemType(eps,EPS_HEP);
	EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);
	EPSSetBalance(eps, EPS_BALANCE_NONE, 0, 0);
	EPSSetType(eps, EPSKRYLOVSCHUR);

	Vec x;
	MatGetVecs(Laplacian,&x,PETSC_NULL);
	VecSet(x,1);
	VecAssemblyBegin(x);
	VecAssemblyEnd(x);
	EPSSetDeflationSpace(eps,1,&x);
	VecDestroy(x);

	ST st;
	EPSGetST(eps,&st);
	STSetType(st,STSINVERT);

	KSP ksp;
	STGetKSP(st,&ksp);
	KSPSetType(ksp,KSPBCGS);

	PC pc;
	KSPGetPC(ksp, &pc);
	PCSetType(pc,PCCHOLESKY);
	KSPSetPC(ksp,pc);

	EPSSetTarget(eps,0.0);

	EPSSetFromOptions(eps);
	EPSSolve(eps);

	PetscScalar algebraicConnectivity;
	EPSGetEigenvalue(eps, 0, &algebraicConnectivity, NULL);

	EPSDestroy(eps);
	MatDestroy(Laplacian);
	SlepcFinalize();

	uint pixelCount = 1;
	for (uint i=0; i<CT->getImgDimension(); ++i)
		pixelCount = pixelCount*(CT->getImgSize()[i]);

	return algebraicConnectivity/(2*(1-cos(PI/ ( round(img[CT->getSortedImgIndices()[pixelCount-1]] ) - round(img[CT->getSortedImgIndices()[0]] ) +1 ) )));

}
