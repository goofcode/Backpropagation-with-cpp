#include "backpropagation.h"

int main( )
{
	//set number of nodes in each layer
	int numnode[NUM_LAYER] = {2, 3, 1};
	Network network(numnode);

	//training set
	double trainingInput[4][2] = {{0,0},{0,1},{1,0},{1,1}};
	//training target set
	double target[4][1] = {{0},{1},{1},{0}};

	network.setRandomWeightConst( );


		
	network.training((double**)trainingInput, (double**)target, 4);
	
}
