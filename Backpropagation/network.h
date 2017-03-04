#pragma once

#define NUM_LAYER 3
#define INITIAL_WEIGHT 0.5
#define INITIAL_CONST 1

#define INPUT_LAYER 0
#define OUTPUT_LAYER NUM_LAYER-1

#define TRAINING_OPP 500000

#define LEARNING_RATE 0.5


class Node
{
private:
	double out;								//out value for node
	double delta;							//delta value for node

	double activationFunc(double input);	

public:
	Node( );
	
	void setOut(double out);
	double getOut( );

	void calculateOut(double net);			//calculate out value from net value

	void setDelta(double delta);
	double getDelta( );
};

class Network
{
private:
	Node **networkNode;			//node[a][b] is b node of a layer 

	int *numNode;               //number of nodes in layers 

	double ***weight;           //weight[a][b][c] is weight from a layer to a+1 layer, from b(a layer) node to c(a+1 layer).
	double **constant;			//constant[a][b] is constant for a+1 layer, b node

	void backpropagation(double *target);
public:
	Network(int *numnode);
	~Network( );
		
	void setRandomWeightConst( );

	void printWeights( );
	void printConstants( );
	void printOutput();

	void calculateOutput(double *input);
	int training(double **trainingInput, double **target, int numCase);
};