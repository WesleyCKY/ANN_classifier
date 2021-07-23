#ifndef ANN_H
#define ANN_H

#include <iostream>
#include <cstdlib> 
#include <chrono>

#include <vector>
#include <string>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <random>
#include <time.h>

#include <omp.h>

using namespace std;

class myANN
{
protected:
	//training config 
	float learningRate;
	int batchSize;
	int stopEpochs;
	int layerNum;
	int nodeNum;

	vector<vector<float>> X;
	vector<float> y;
	
	vector<float> weight;
	vector<int> predict;
	
	vector< vector<float> > firstWeight;
	vector < vector < vector<float> >> innerWeights;
	vector< vector<float> > lastWeight;
	
	vector<float> firstBias;
	vector< vector<float> > innerBiases;
	vector<float> lastBias;
	
	vector< vector<float> > hiddenLayer;
	vector < vector < vector<float> >> hiddenLayer_F;
	
	vector < vector < vector<float> >> dataSet;
	vector <vector<float> > ansSet;
	
	vector <vector<float> > Output_feedforward;
	vector <vector<float> > errorOfHiddenLayer;
public:
	myANN();

	void debug(vector< vector<float> > X);

	//1: feedforward
	void weightConfig(vector< vector<float>> X, int numOfHiddenLayer, int numOfNode);
	vector<int> feedForward(vector< vector<float>> X);
	
	//2: train function: to train the model on a given data set. Using mini-batch SGD and backpropagation algorithm for the training
	void trainConfig(float r, int b, int s);
	void train(vector< vector<float> > X, vector<float> y);
	
	//3: inference function: given an input vector x, predict its corresponding digit
	void inference(vector<float> X);

	//4: weight store/load functions: to store or load the weight values to or from local disk file
	void saveWeight(string fileName);
	void loadWeight(string fileName);
	int threadNum = 2; //2, 4, 8, 16

private:
	//generate random weight
	float randomWeight();
	
	//define activation function --> sigmoid
	float sigmoid(float input);
	float dsigmoid(float in);

	vector<float> scalarV(vector<float> V, float Snum);
	vector < vector<float>> matrixTranspose(vector<vector<float>>& m);
	vector < vector<float>> matrixTimesMatrix(vector< vector<float> > X, vector< vector<float> > Y);
	vector<float> matrixTimesVector(vector< vector<float> > X, vector<float> Y);
	vector<float> vectorTimesVector(vector<float> V1, vector<float> V2);
	vector<vector<float> >vectorTimesVectorOutMatrix(vector<float> V1, vector<float> V2);
};

void myANN::debug(vector< vector<float> > X) {
	cout << "------------testing--------------" << endl;

	vector<float> V;
	V.resize(X.size());

	for (int i = 0; i < 500; i++) {
		V = matrixTimesVector(firstWeight, X[0]);
	}

	//display
	for (int i = 0; i < V.size(); i++)
	{
		cout << V[i] << "  ";
	}
	cout << "------------testing--------------" << endl;
}

myANN::myANN() {
	
}

// define sigmoid function
float myANN::sigmoid(float toSigmoid) {
	return (1 / (1 + exp(-toSigmoid)));
}

// define dsigmoid function
float myANN::dsigmoid(float Sig) {
	return Sig * (1 - Sig);
}

//define how to generate random weights
float myANN::randomWeight() {
	return (2 * (float)rand() / (RAND_MAX)-1);;
}

vector<float> myANN::scalarV(vector<float> V, float Snum) {
	for (int i = 0; i < V.size(); i++) {
		V[i] = V[i] * Snum;
		//cout << V[i];
	}
	return V;
}

void myANN::trainConfig(float r, int b, int s) {
	this->learningRate = r;
	this->batchSize = b;
	this->stopEpochs = s;
}

void myANN::weightConfig(vector< vector<float> > X, int numOfHiddenLayer, int numOfNode) {
	this->layerNum = numOfHiddenLayer;
	this->nodeNum = numOfNode;

	firstWeight.resize(X.size(), vector<float>(X[0].size()));
	firstBias.resize(X[0].size());
	innerWeights.resize((layerNum - 1), vector <vector<float>>(nodeNum, vector<float>(nodeNum)));
	innerBiases.resize((layerNum - 1), vector<float>(nodeNum));
	lastWeight.resize(10, vector<float>(nodeNum));
	lastBias.resize(10);
	hiddenLayer.resize(layerNum, vector<float>(nodeNum));
	hiddenLayer_F.resize(batchSize, vector <vector<float>>(layerNum, vector<float>(nodeNum)));

	//Weight after input
	for (int i = 0; i < X.size(); i++) {
		for (int j = 0; j < X[0].size(); j++)
		{
			firstWeight[i][j] = randomWeight();
		}
	}

	//Bias after input
	for (int i = 0; i < X[0].size(); i++)
	{
		firstBias[i] = randomWeight();
	}

	//Weight after hiddenLayer1 to hiddenLayer Last
	for (int i = 0; i < layerNum - 1; i++) {
		for (int j = 0; j < nodeNum; j++) {
			for (int k = 0; k < nodeNum; k++)
			{
				innerWeights[i][j][k] = randomWeight();
			}
		}
	}

	//weight for output
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < nodeNum; j++)
		{
			lastWeight[i][j] = randomWeight();
		}
	}

	/*for (int i = 0; i < layerNum; i++) testing(otherWeight[i]);*/

	//Bias after hiddenLayer1 to hiddenLayer Last
	for (int i = 0; i < layerNum - 1; i++) {
		for (int j = 0; j < nodeNum; j++)
		{
			innerBiases[i][j] = randomWeight();
		}
	}
	for (int i = 0; i < 10; i++)
	{
		lastBias[i] = randomWeight();
	}
	//testing(otherBias);
}




//vector multipication (vector)
vector<float> myANN::vectorTimesVector(vector<float> V1, vector<float> V2) { 
	vector<float> Mat;
	Mat.resize(V1.size());
	int i;
//#pragma omp parallel for num_threads(threadNum) \
//	default(none) shared(V1, V2, Mat) private(i)
	for (i = 0; i < V1.size(); i++) {
		//for (int j = 0; j < V2.size(); j++) {
		Mat[i] = V1[i] * V2[i];
		//}
	}
	return Mat;
}

//vector multipication --> output matrix
vector<vector<float> > myANN::vectorTimesVectorOutMatrix(vector<float> V1, vector<float> V2) { 
	vector<vector<float>> Mat;
	Mat.resize(V1.size(), vector<float>(V2.size()));
	int i, j;

#pragma omp parallel for num_threads(threadNum) \
	   default(none) private(i,j) shared(V1, V2, Mat)
	for (i = 0; i < V1.size(); i++) {
		for (j = 0; j < V2.size(); j++) {
			Mat[i][j] = V1[i] * V2[j];
		}
	}
	return Mat;
}

//matrix transpose
vector < vector<float>> myANN::matrixTranspose(vector<vector<float> >& m) {  
	vector<vector<float>> matrixTrans(m[0].size(), vector<float>());

	for (int i = 0; i < m.size(); i++)
	{
		for (int j = 0; j < m[i].size(); j++)
		{
			matrixTrans[j].push_back(m[i][j]);
		}
	}

	return matrixTrans;
}


//matrix matrix multiplication
vector < vector<float>> myANN::matrixTimesMatrix(vector< vector<float> > A, vector< vector<float> > B) 
{
	float a = 0;
	vector< vector<float> > outputMatrix;
	outputMatrix.resize(A.size(), vector<float>(B[0].size()));
	int i, j, k;
#  pragma omp parallel for num_threads(threadNum) \
	   default(none) private(i, j, k, a, outputMatrix) shared(A, B)
	//private(i,j,k)
	for (i = 0; i < A.size(); i++) {
		for (j = 0; j < B[0].size(); j++) {
			for (k = 0; k < B.size(); k++) {
				a = a + (A[i][k] * B[k][j]);

			}
			outputMatrix[i][j] = a;
			a = 0;
		}
	}
	return outputMatrix;
}

vector<float> myANN::matrixTimesVector(vector< vector<float> > X, vector<float> Y)
{
	vector<float> V;
	V.resize(X.size());
	int i, j;
	double sum;
#pragma omp parallel for num_threads(threadNum) \
	   default(none) private(sum, i,j) shared(V, X, Y)
	for (i = 0; i < X.size(); i++)
	{
		sum = 0;
		for (j = 0; j < X[0].size(); j++) {
			sum += (X[i][j]) * (Y[j]);
		}
		V[i] = sum;
	}

	return V;
}


// carry out the layer-wise feedforward calculations
vector<int>  myANN::feedForward(vector< vector<float> > X) {
	
	vector<float> outPut;
	outPut.resize(10);
	predict.resize(X.size());
	Output_feedforward.resize(X.size(), vector<float>(10));

	for (int allData = 0; allData < X.size(); allData++) {
		hiddenLayer_F[allData][0] = matrixTimesVector(firstWeight, X[allData]);
		for (int i = 0; i < hiddenLayer_F[allData][0].size(); i++) {
			hiddenLayer_F[allData][0][i] = (sigmoid(hiddenLayer_F[allData][0][i] + firstBias[i]));
		}

		//testing(hiddenLayer);

		// hidden layer[1] to the last...i
		for (int i = 1; i < layerNum; i++) {
			hiddenLayer_F[allData][i] = matrixTimesVector(innerWeights[(i - 1)], hiddenLayer_F[allData][i - 1]);
			for (int j = 0; j < nodeNum; j++) {
				hiddenLayer_F[allData][i][j] = (sigmoid(hiddenLayer_F[allData][i][j] + innerBiases[i - 1][j]));
			}
			//testing(otherWeight[i - 1]);
		}


		//Generate the output and predict

		outPut = matrixTimesVector(lastWeight, hiddenLayer_F[allData][(layerNum - 1)]);

		predict[allData] = 0;
		float pre = sigmoid(outPut[0] + lastBias[0]);

		for (int i = 0; i < outPut.size(); i++) {

			outPut[i] = sigmoid(outPut[i] + lastBias[i]);

			// store last layer output for backpropagation
			Output_feedforward[allData][i] = outPut[i];

			//prediction in Feed Forward
			if (outPut[i] > pre) {
				pre = outPut[i];
				predict[allData] = i;
			}

		}


	}
	return predict;
}


//use mini-batch SGD and backpropagation algorithm for the training
void myANN::train(vector< vector<float> > X, vector<float> y) {
	vector<vector<float>> forward;
	clock_t epoStart, epoEnd;
	int numThread = 16;
	clock_t batchStart, batchEnd;
	//Mini-batch SGD algorithm

	for (int t = 0; t < stopEpochs; t++) {
		vector<int> predict;

		//cout << endl;
		cout << "Epochs: " << t + 1 << endl;
		//cout << "=========================================================================================" << endl;
		//cout << "-----------------------------------------------------------------------------------------" << endl;
		//cout << "........................................................................................." << endl;
		
		epoStart = clock();
		/*S = D/B mini-batches
		random shuffle and divide it into mini-batches d1,d2,d3,...,ds*/
		
		//suffle by time
		/*unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		shuffle(X.begin(), X.end(), default_random_engine(seed));
		shuffle(y.begin(), y.end(), default_random_engine(seed));*/


		//suffle num from 1 to 10
		int ran = (rand() % 10) + 1;
		shuffle(X.begin(), X.end(), default_random_engine(ran));
		shuffle(y.begin(), y.end(), default_random_engine(ran));


		dataSet.resize((X.size() / batchSize), vector <vector<float>>(batchSize, vector<float>(X[0].size())));
		ansSet.resize((X.size() / batchSize), vector<float>(batchSize));


		for (int i = 0; i < (X.size() / batchSize); i++) {
			
			for (int j = 0; j < batchSize; j++) {
				for (int k = 0; k < X[0].size(); k++) {
					dataSet[i][j][k] = X[(j + (i * batchSize))][k];
				}
				ansSet[i][j] = y[j + (i * batchSize)];
			}
		}

		for (int set = 0; set < (X.size() / batchSize); set++) {
			batchStart = clock();
			//predict and calculate accuracies
			forward.resize(nodeNum, vector<float>(dataSet[set][0].size()));
			//cout << endl;
			cout << "Dataset:" << set + 1 << endl;
			
			
			//cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
			//cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;

			predict = feedForward(dataSet[set]);

			int correct = 0;
			for (int i = 0; i < predict.size(); i++) {
				if (predict[i] == ansSet[set][i]) {
					correct++;
				}
			}

			double Accuracy = (double)correct / predict.size();
			//cout << "Accuracy : " << Accuracy * 100 << "%" << endl;

			vector<float> avgError;
			float Ferror;
			avgError.resize(10);

			for (int i = 0; i < batchSize; i++) {
				//cout << "batch= " << i << endl << endl;
				for (int j = 0; j < 10; j++) {
					//cout << "j = " << j << endl;
					if (j == ansSet[set][i])
					{
						avgError[j] += (0.5 * pow((1 - Output_feedforward[i][j]), 2));
						//cout << "j == ansSet[set][i] --> error:" << Output_feedforward[i][j] << endl;
					}
					else {
						avgError[j] += (0.5 * pow((0 - Output_feedforward[i][j]), 2));
						//cout << "j != ansSet[set][i] --> error:"<< Output_feedforward[i][j] << endl;
					}
				}
			}

			// show average error in batched dataset
			for (int i = 0; i < 10; i++) {

				avgError[i] = avgError[i] / 32;
				//cout << "average error of " << i << ": " << avgError[i] << endl;
			}

			//find out error of the other layer 
			for (int bSize = 0; bSize < batchSize; bSize++) {

				errorOfHiddenLayer.resize((layerNum - 1), vector<float>(nodeNum));

				//The last hidden layer to output
				vector<float> DeltaWeightOut;
				vector<vector<float>> dwo;
				vector<float> ErrorOfFirstHiddenLayer; // Error
				vector<vector<float>> HiddenError; // Error
				
				for (int i = 0; i < Output_feedforward[bSize].size(); i++) 
				{
					Output_feedforward[bSize][i] = (dsigmoid(Output_feedforward[bSize][i]));
				}
				
				
				DeltaWeightOut = vectorTimesVector(avgError, Output_feedforward[bSize]);
				DeltaWeightOut = scalarV(DeltaWeightOut, learningRate);
				dwo = vectorTimesVectorOutMatrix(DeltaWeightOut, hiddenLayer_F[bSize][layerNum - 1]);

				//change the weight of last hidden layer to output
				for (int i = 0; i < 10; i++) {
					for (int j = 0; j < nodeNum; j++) {
						lastWeight[i][j] = lastWeight[i][j] - dwo[i][j];
					}
				}

				vector<float> DeltaH;
				vector<vector<float>> DeltaH_M;
				HiddenError.resize((layerNum - 1), vector<float>(nodeNum));

				ErrorOfFirstHiddenLayer = matrixTimesVector(matrixTranspose(lastWeight), avgError);
				if (layerNum > 1) {
					for (int i = 0; i < hiddenLayer_F[bSize].size(); i++) {
						for (int n = 0; n < hiddenLayer_F[bSize][i].size(); n++) {
							hiddenLayer_F[bSize][i][n] = (dsigmoid(hiddenLayer_F[bSize][i][n]));
						}
					}

					for (int i = 1; i < layerNum; i++) {
						if (i == 1) {
							HiddenError[i - 1] = matrixTimesVector(matrixTranspose(innerWeights[layerNum - i - 1]), ErrorOfFirstHiddenLayer);
							DeltaH = vectorTimesVector(ErrorOfFirstHiddenLayer, hiddenLayer_F[bSize][i]);
							DeltaH = scalarV(DeltaH, learningRate);
							DeltaH_M = vectorTimesVectorOutMatrix(DeltaH, firstWeight[bSize]);
							for (int a = 0; a < nodeNum; a++) {
								for (int b = 0; b < firstWeight[0].size(); b++) {
									firstWeight[a][b] = firstWeight[a][b] - DeltaH_M[a][b];
								}
							}
						}
						else {
							HiddenError[i - 1] = matrixTimesVector(matrixTranspose(innerWeights[layerNum - i - 1]), hiddenLayer_F[bSize][layerNum - i]);
							DeltaH = vectorTimesVector(HiddenError[i - 1], hiddenLayer_F[bSize][i - 1]);
							DeltaH = scalarV(DeltaH, learningRate);
							DeltaH_M = vectorTimesVectorOutMatrix(DeltaH, hiddenLayer_F[bSize][layerNum - 1 - i]);

							for (int a = 0; a < nodeNum; a++) {
								for (int b = 0; b < innerWeights[layerNum - i - 1].size(); b++) {
									innerWeights[layerNum - i - 1][a][b] = innerWeights[layerNum - 1 - i][a][b] - DeltaH_M[a][b];
								}
							}
						}

					}
				}
				else { //change the weight between first input to first hidden layer
					vector<float> DeltaIn;
					vector<vector<float>> DeltaIn_M;
					vector<float> DWIBias;
					ErrorOfFirstHiddenLayer.resize(10);
					ErrorOfFirstHiddenLayer = matrixTimesVector(matrixTranspose(lastWeight), avgError);
					for (int i = 0; i < hiddenLayer_F[bSize][0].size(); i++) {		//disigmoid the whole
						hiddenLayer_F[bSize][0][i] = (dsigmoid(hiddenLayer_F[bSize][0][i]));
					}

					DeltaIn = vectorTimesVector(ErrorOfFirstHiddenLayer, Output_feedforward[bSize]);
					DeltaIn = scalarV(DeltaIn, learningRate);
					DeltaIn_M = vectorTimesVectorOutMatrix(DeltaIn, firstWeight[bSize]);


					//cout << firstWeight.size() << firstWeight[0].size() << DeltaIn_M.size() << DeltaIn_M[0].size()<< endl;
					for (int i = 0; i < nodeNum; i++) {
						for (int j = 0; j < firstWeight[0].size(); j++) {
							//cout << firstWeight.size() << firstWeight[0].size() << DeltaIn_M.size() << DeltaIn_M[0].size() << endl;
							firstWeight[i][j] = firstWeight[i][j] + DeltaIn_M[i][j];
						}
						//cout << i << endl;
					}

				}
			}
			batchEnd = clock() - batchStart;
			cout << "Batch time:" << (float)batchEnd / CLOCKS_PER_SEC << endl;
		}
		epoEnd = clock() - epoStart;
		cout << "Epoch time:" << (float)epoEnd / CLOCKS_PER_SEC << endl;
	}

}

/*   inference function: given an input vector x, predict its corresponding digit.  */
void myANN::inference(vector<float> X) {
	vector<float> outPut;
	outPut.resize(10);

	hiddenLayer[0] = matrixTimesVector(firstWeight, X);
	for (int i = 0; i < hiddenLayer[0].size(); i++) {
		hiddenLayer[0][i] = (sigmoid(hiddenLayer[0][i] + firstBias[i]));
	}

	//hidden layer[1] to the last...
	for (int i = 1; i < layerNum; i++) {
		hiddenLayer[i] = matrixTimesVector(innerWeights[(i - 1)], hiddenLayer[i - 1]);
		for (int j = 0; j < nodeNum; j++) {
			hiddenLayer[i][j] = (sigmoid(hiddenLayer[i][j] + innerBiases[i - 1][j]));
		}
	}

	outPut = matrixTimesVector(lastWeight, hiddenLayer[(layerNum - 1)]);

	int preNum = 0;
	float preValue = 0;

	for (int i = 0; i < outPut.size(); i++) {
		outPut[i] = (sigmoid(outPut[i] + lastBias[i]));

		if (outPut[i] > preValue) {
			preValue = outPut[i];
			preNum = i;
		}
	}
	cout << "prediction : " << preNum;
}

void myANN::saveWeight(string fileName) {
	ofstream outFile;
	outFile.open(fileName);

	for (int i = 0; i < firstWeight.size(); i++)
	{
		for (int j = 0; j < firstWeight[0].size(); j++)
		{
			outFile << firstWeight[i][j] << endl;
		}
	}

	for (int i = 0; i < innerWeights.size(); i++)
	{
		for (int j = 0; j < innerWeights[0].size(); j++)
		{
			for (int k = 0; k < innerWeights[0][0].size(); k++)
			{
				outFile << innerWeights[i][j][k] << endl;
			}
		}
	}

	for (int i = 0; i < lastWeight.size(); i++)
	{
		for (int j = 0; j < lastWeight[0].size(); j++)
		{
			outFile << lastWeight[i][j] << endl;
		}
	}

	outFile.close();
	cout << "weight file: " << fileName << " saved." << endl;

}

void myANN::loadWeight(string fileName) {
	ifstream inFile;
	inFile.open(fileName);

	for (int i = 0; i < firstWeight.size(); i++)
	{
		for (int j = 0; j < firstWeight[0].size(); j++)
		{
			inFile >> firstWeight[i][j];
		}
	}

	for (int i = 0; i < innerWeights.size(); i++)
	{
		for (int j = 0; j < innerWeights[0].size(); j++)
		{
			for (int k = 0; k < innerWeights[0][0].size(); k++)
			{
				inFile >> innerWeights[i][j][k];
			}
		}
	}

	for (int i = 0; i < lastWeight.size(); i++)
	{
		for (int j = 0; j < lastWeight[0].size(); j++)
		{
			inFile >> lastWeight[i][j];
		}
	}

	cout << "weight file: " << fileName <<" loaded."<<endl;

}


#endif