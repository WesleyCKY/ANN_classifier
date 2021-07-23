#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <time.h>
#include <windows.h>
#include <algorithm>
#include "myANN.h"
//#include "OpenMPANN.h"

#include <omp.h>

using namespace std;

int main(int argc, const char* argv[]) {

	vector< vector<float> > train_X; //matrix
	vector<float> train_Y;			 //vector

	//ifstream trainFile("train_small.txt");
	ifstream trainFile("train.txt");

	if (trainFile.is_open())
	{
		cout << "Loading data ..." << endl;
		string line;
		while (getline(trainFile, line))
		{
			int x, y;
			vector<float> X;
			stringstream ss(line);
			ss >> y;
			train_Y.push_back(y);
			for (int i = 0; i < 28 * 28; i++) {
				ss >> x;
				X.push_back(x / 255.0);
			}
			train_X.push_back(X);
		}

		trainFile.close();
		cout << "Loading data finished !" << endl;
	}
	else
		cout << "Cannot open file !" << endl;

	//train class
	myANN ANN;

	clock_t start, end;

	//Config: 1:learning rate, 2:batch size, 3:number of epochs
	ANN.trainConfig(0.1, 32, 3);

	//Config: 1:input, 2:hidden layer numbers, 3:node numbers
	ANN.weightConfig(train_X, 2, 5);

	//Load Weight
	ANN.loadWeight("weight1.txt");

	//Start
	start = clock();
	ANN.train(train_X, train_Y);
	end = clock() - start;

	cout << "single thread CPU Time: " << (float)end / CLOCKS_PER_SEC << endl;

	return 0;
}