#include "network.h"
#include <cstdlib>
#include <ctime>
#include <iostream>


//#define __WEIGHT_DEBUG__
//#define __TRAINING_DEBUG__

using std::cout;
using std::endl;

Node::Node( )
{
	out = 0; delta = 0;
}
double Node::activationFunc(double input)
{
	//sigmoid function
	return 1 / (1 + exp(-input));
}

void Node::setOut(double out)
{
	this->out = out;
}
double Node::getOut( )
{
	return out;
}

void Node::calculateOut(double net)
{
	out = activationFunc(net);
}

void Node::setDelta(double delta )
{
	this->delta = delta;
}
double Node::getDelta( )
{
	return delta;
}

Network::Network(int *numnode)
{
	numNode = new int[NUM_LAYER];

	networkNode = new Node*[NUM_LAYER];
	
	weight = new double**[NUM_LAYER - 1];
	constant = new double*[NUM_LAYER - 1];

	for(int i = 0; i<NUM_LAYER; i++)
	{
		numNode[i] = numnode[i];
		networkNode[i] = new Node[numNode[i]];
	}

	for(int i = 0; i<NUM_LAYER - 1; i++)
	{
		weight[i] = new double*[numNode[i]];
		constant[i] = new double[numNode[i + 1]];

		for(int j = 0; j<numNode[i]; j++)
		{
			weight[i][j] = new double[numNode[i + 1]];
			for(int k = 0; k<numNode[i + 1]; k++)
				weight[i][j][k] = INITIAL_WEIGHT;
		}

		for(int j = 0; j < numNode[i + 1]; j++)
			constant[i][j] = INITIAL_CONST;
	}
	srand(time(NULL));
}
Network::~Network( )
{
	for(int i = 0; i < NUM_LAYER; i++)
		delete networkNode[i];
	delete networkNode;

	for(int i = 0; i < NUM_LAYER - 1; i++)
	{
		for(int j = 0; j < numNode[i]; j++)
			delete weight[i][j];
		delete weight[i];

		delete constant[i];
	}

	delete weight;
	delete constant;
	delete numNode;
}

void Network::setRandomWeightConst( )
{
	for(int i = 0; i<NUM_LAYER - 1; i++)
	{
		for(int j = 0; j < numNode[i]; j++)
			for(int k = 0; k < numNode[i + 1]; k++)
				weight[i][j][k] = (rand( ) % 99 + 1) / 100.0;

		for(int j = 0; j < numNode[i + 1]; j++)
			constant[i][j] = INITIAL_CONST;//(rand( ) % 99 + 1) / 100.0;
	}
}

void Network::printWeights( )
{
	for(int i = 0; i<NUM_LAYER -1; i++)
	{
		cout << "Layer " << i + 1 << " -> Layer " << i + 2 << endl;

		for(int j = 0; j<numNode[i]; j++)
		{
			cout << "node " << j << ": ";

			for(int k = 0; k<numNode[i + 1]; k++)
				cout << weight[i][j][k]<<" ";
			cout << endl;
		}
		cout << endl;
	}
}
void Network::printConstants( )
{
	cout<<"constant: "<<endl;
	
	for(int i = 0; i < NUM_LAYER - 1; i++)
	{
		cout<< "Layer " << i + 2 << endl;
		
		for(int j = 0; j < numNode[i + 1]; j++)
			cout <<"\t"<< j + 1 << " node: " << constant[i][j] << endl;
		cout << endl;
	}
}
void Network::printOutput()
{
	cout<<"output: ";
	
	for(int i=0;i<numNode[OUTPUT_LAYER];i++)
		cout<<networkNode[OUTPUT_LAYER][i].getOut()<<" ";
	cout<<endl;
}

void Network::calculateOutput(double *input)
{
	double net;

	//input setting
	for(int i = 0; i < numNode[INPUT_LAYER]; i++)
		networkNode[INPUT_LAYER][i].setOut(input[i]);

	//processing
#if defined __WEIGHT_DEBUG__	
	//processing
	cout << "Layer 0" << endl;
	cout << "\tinput: ";
	for(int i = 0; i < numNode[INPUT_LAYER]; i++)
		cout << input[i] << " ";
	cout << endl << endl;

	for(int i = 0; i < NUM_LAYER - 1; i++)
	{
		cout << "Layer " << i + 1 << endl;

		for(int k = 0; k < numNode[i + 1]; k++)
		{
			cout << "\t" << "node "<<k << endl;

			net = 0;

			cout << "\tnet = 0" << endl;

			for(int j = 0; j < numNode[i]; j++)
			{
				net += weight[i][j][k] * networkNode[i][j].getOut( );
				cout << "\tnet += " << weight[i][j][k]<<" * "<<networkNode[i][j].getOut( ) <<" = "<< weight[i][j][k] * networkNode[i][j].getOut( ) << endl;
			}

			net += constant[i][k];
			cout << "\tnet += const " << constant[i][k] << endl << endl;

			networkNode[i + 1][k].calculateOut(net);

			cout << "\t\tnet result: " << net << endl;
			cout << "\t\tout: " << networkNode[i + 1][k].getOut( ) << endl << endl;
		}
	}

#elif defined __TRAINING_DEBUG__
	
	cout << "\t\tInput: "; 
	for(int i = 0; i < numNode[INPUT_LAYER]; i++)
		cout << networkNode[0][i].getOut( ) << " ";
	
	for(int i = 0; i < NUM_LAYER - 1; i++)
	{
		for(int k = 0; k < numNode[i + 1]; k++)
		{
			net = 0;

			for(int j = 0; j < numNode[i]; j++)
				net += weight[i][j][k] * networkNode[i][j].getOut( );

			net += constant[i][k];

			networkNode[i + 1][k].calculateOut(net);
		}
	}
	cout << "  output:";
	for(int i = 0; i < numNode[OUTPUT_LAYER]; i++)
		cout << networkNode[OUTPUT_LAYER][i].getOut( )<<" ";
	
#else
	for(int i = 0; i < NUM_LAYER - 1; i++)
	{
		for(int k = 0; k < numNode[i + 1]; k++)
		{
			net = 0;

			for(int j = 0; j < numNode[i]; j++)
				net += weight[i][j][k] * networkNode[i][j].getOut( );

			net += constant[i][k];

			networkNode[i + 1][k].calculateOut(net);
		}
	}
#endif

}


int Network::training(double **trainingInput, double **target, int numCase)
{

#ifdef __TRAINING_DEBUG__
//function training for training debug
	double *caseInput, *caseTarget;
	for(int i = 0; i < TRAINING_OPP; i++)
	{
		cout << "training number " << i << endl;
	
		for(int j = 0; j < numCase; j++)
		{
			caseInput = (double*)(trainingInput + 2 * j * numNode[INPUT_LAYER]);
			caseTarget = (double*)(target + 2 * j * numNode[OUTPUT_LAYER]);

			cout << "\ttraining case " << j << endl;

			//calculate befor adjustment
			calculateOutput(caseInput);

			cout << "   traget: ";
			for(int k = 0; k < numNode[OUTPUT_LAYER]; k++)
				cout << caseTarget[k]<< " ";
			cout<<endl<<endl;
			

			//*******backpropagation*******//
			backpropagation(caseTarget);

			cout << endl;

			//calculate after adjustment
			calculateOutput(caseInput);

			cout << "   traget: ";
			for(int k = 0; k < numNode[OUTPUT_LAYER]; k++)
				cout << caseTarget[k] << " ";
			cout << endl << endl;

			//getchar( );
			
		}
		cout<<endl;
	}
	cout << "training done!" << endl << "number of execution" << TRAINING_OPP << endl;

	return 1;
#endif

#ifndef __TRAINING_DEBUG__
//function training for normal debug
	double *caseInput, *caseTarget;
	
	cout << "before training" << endl;
	for(int i = 0; i < TRAINING_OPP; i++)
	{
		if(i == TRAINING_OPP - 1)
			cout << endl << "training done!" << endl << "number of execution " << TRAINING_OPP << endl << endl;

		for(int j = 0; j < numCase; j++)
		{
			caseInput = (double*)(trainingInput + 2 * j * numNode[INPUT_LAYER]);
			caseTarget = (double*)(target + 2 * j * numNode[OUTPUT_LAYER]);

			//calculate befor adjustment
			calculateOutput(caseInput);
			if(i == 0)
			{
				cout << "\tinput: ";
				for(int k = 0; k < numNode[INPUT_LAYER]; k++)
					cout << caseInput[k] << " ";
				cout << "  ";
				printOutput( );
			}

			//*******backpropagation*******//
			backpropagation(caseTarget);

			if(i == TRAINING_OPP-1)
			{
				calculateOutput(caseInput);
				cout << "\tinput: ";
				for(int k = 0; k < numNode[INPUT_LAYER]; k++)
					cout << caseInput[k] << " ";
				cout << "  ";
				printOutput( );
			}
		}
	}

	return 1;
#endif
}

void Network::backpropagation(double *target)
{
#ifndef __TRAINING_DEBUG__
	double out;
	double subsum;

	//hidden layer -> output layer
	for(int i = 0; i < numNode[OUTPUT_LAYER]; i++)
	{
		//delta value setting
		out = networkNode[OUTPUT_LAYER][i].getOut( );
		networkNode[OUTPUT_LAYER][i].setDelta(out*(1 - out)*(target[i] - out));
	}

	for(int i = 0; i < numNode[OUTPUT_LAYER - 1]; i++)
		//weight adjustment
		for(int j = 0; j < numNode[OUTPUT_LAYER]; j++)
			weight[OUTPUT_LAYER - 1][i][j] += LEARNING_RATE * networkNode[OUTPUT_LAYER][j].getDelta( ) * networkNode[OUTPUT_LAYER - 1][i].getOut( );

	//hidden layer -> hidden layer OR input layer -> hidden layer
	for(int i = OUTPUT_LAYER - 1; i >= 1; i--)
	{
		//delta value setting
		for(int j = 0; j < numNode[i]; j++)
		{
			subsum = 0;
			out = networkNode[i][j].getOut( );
			for(int k = 0; k < numNode[i + 1]; k++)
				subsum += weight[i][j][k] * networkNode[i + 1][k].getDelta( );
			networkNode[i][j].setDelta(out*(1 - out)*subsum);
		}
		//weight adjustment
		for(int j = 0; j < numNode[i - 1]; j++)
			for(int k = 0; k < numNode[i]; k++)
				weight[i - 1][j][k] += LEARNING_RATE * networkNode[i][k].getDelta( ) * networkNode[i - 1][j].getOut( );
	}

#endif

#ifdef __TRAINING_DEBUG__
	double out;
	double subsum;
	double temp;

	//hidden layer -> output layer
	for(int i = 0; i < numNode[OUTPUT_LAYER]; i++)
	{
		//delta value setting
		out = networkNode[OUTPUT_LAYER][i].getOut( );
		networkNode[OUTPUT_LAYER][i].setDelta(out*(1 - out)*(target[i] - out));
	}
	
	cout << "\t\tlayer " << OUTPUT_LAYER - 1 << " -> " << OUTPUT_LAYER << endl;
	for(int i = 0; i < numNode[OUTPUT_LAYER - 1]; i++)
	{
		//weight adjustment
		for(int j = 0; j < numNode[OUTPUT_LAYER]; j++)
		{
			temp = LEARNING_RATE * networkNode[OUTPUT_LAYER][j].getDelta( ) * networkNode[OUTPUT_LAYER - 1][i].getOut( );
			weight[OUTPUT_LAYER - 1][i][j] += temp;
			cout << "\t\t\tnode " << i << " -> " << j << "   weight += " << temp << endl;
		}
	}
	//hidden layer -> hidden layer OR input layer -> hidden layer
	for(int i = OUTPUT_LAYER - 1; i >= 1; i--)
	{
		//delta value setting
		for(int j = 0; j < numNode[i]; j++)
		{
			subsum = 0;
			out = networkNode[i][j].getOut( );
			for(int k = 0; k < numNode[i + 1]; k++)
				subsum += weight[i][j][k] * networkNode[i + 1][k].getDelta( );
			networkNode[i][j].setDelta(out*(1 - out)*subsum);
		}
		//weight adjustment
		cout << "\t\tlayer " << i-1 << " -> " << i << endl;
		for(int j = 0; j < numNode[i - 1]; j++)
			for(int k = 0; k < numNode[i]; k++)
			{
				temp = LEARNING_RATE * networkNode[i][k].getDelta( ) * networkNode[i - 1][j].getOut( );
				weight[i - 1][j][k] += temp;
				cout << "\t\t\tnode " << j << " -> " << k << "   weight += " << temp << endl;
			}
	}
#endif
}