
//Hidden Markov Model, Random Restarts, Multi-Threaded, PThread

#include <stdio.h>
#include <pthread.h>

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <fstream>

#include <math.h> 
#include <limits>
#include <float.h>

#include <algorithm>
#include <set>
#include <map>
#include <sstream>

using std::string;
using std::cout;
using std::vector;

using std::map;
using std::tuple;

class HMM
{
	vector<int> obs;
	vector<string> alphabet;

	int N;
	int M;
	int T;

	vector<double> c;
	vector<vector<double>> alpha;
	vector<vector<double>> beta;
	vector<vector<double>> gamma;
	vector<vector<vector<double>>> digamma;

	int minIters = 30;
	int maxIters = 200;
	double epsilon = 0.0000001;
	int iters = 0;
	double oldLogProb;// = -DBL_MAX;

	std::mt19937 rand;

	double randNextDouble();

	int seed;

public:
	double logProb;

	vector<vector<double>> a;
	vector<vector<double>> b;
	vector<double> pi;

	void AlphaPass(vector<int> obsSeq);
	void BetaPass(vector<int> obsSeq);
	void GammaPass(vector<int> obsSeq);
	void ReestimateModel(vector<int> obsSeq);
	void ComputeLog(vector<int> obsSeq);
	void PrintMatrices();

	void RandInitMatrices();
	void Run();

	HMM();
	HMM(vector<int> _obs, vector<string> _alphabet, int _N, int seed);

	void init(vector<int> _obs, vector<string> _alphabet, int _N, int seed);


};

HMM::HMM()
{
}

HMM::HMM(vector<int> _obs, vector<string> _alphabet, int _N, int seed)
{
	init(_obs, _alphabet, _N, seed);
}

void HMM::init(vector<int> _obs, vector<string> _alphabet, int _N, int _seed)
{
	//oldLogProb = std::numeric_limits<double>::min();
	oldLogProb = -DBL_MAX;

	seed = _seed;
	rand.seed(seed);

	obs = _obs;
	alphabet = _alphabet;

	N = _N;
	M = alphabet.size();
	T = obs.size();


	//resize vectors
	/*
	double a[N][N];
	double b[N][M];
	double pi[N];

	double c[T];
	double alpha[T][N];
	double beta[T][N];
	double gamma[T][N];
	double digamma[T][N][N];
	*/

	a.resize(N);
	for (unsigned i = 0; i < a.size(); i++)
	{
		a[i].resize(N);
	}

	b.resize(N);
	for (unsigned i = 0; i < b.size(); i++)
	{
		b[i].resize(M);
	}

	pi.resize(N);

	c.resize(T);

	alpha.resize(T);
	for (unsigned i = 0; i < alpha.size(); i++)
	{
		alpha[i].resize(N);
	}

	beta.resize(T);
	for (unsigned i = 0; i < beta.size(); i++)
	{
		beta[i].resize(N);
	}

	gamma.resize(T);
	for (unsigned i = 0; i < gamma.size(); i++)
	{
		gamma[i].resize(N);
	}

	digamma.resize(T);
	for (unsigned i = 0; i < digamma.size(); i++)
	{
		digamma[i].resize(N);
		for (unsigned j = 0; j < digamma[i].size(); j++)
		{
			digamma[i][j].resize(N);
		}
	}
}

double HMM::randNextDouble()
{
	return ((double)rand()) / ((double)rand.max());
}

void HMM::AlphaPass(vector<int> obsSeq)
{
	//compute alpha[0,i]
	c[0] = 0.0;
	for (int i = 0; i < N; i++)
	{
		alpha[0][i] = pi[i] * b[i][obsSeq[0]];
		c[0] = c[0] + alpha[0][i];
	}

	//scale the alpha[0,i]
	c[0] = 1 / c[0];
	for (int i = 0; i < N; i++)
	{
		alpha[0][i] = c[0] * alpha[0][i];
	}

	//compute alpha[t,i]
	for (int t = 1; t < T; t++)
	{
		c[t] = 0;
		for (int i = 0; i < N; i++)
		{
			alpha[t][i] = 0;
			for (int j = 0; j < N; j++)
			{
				alpha[t][i] += alpha[t - 1][j] * a[j][i];
			}
			alpha[t][i] *= b[i][obsSeq[t]];
			c[t] += alpha[t][i];
		}

		//scale alpha[t,i]
		c[t] = 1 / c[t];
		for (int i = 0; i < N; i++)
		{
			alpha[t][i] *= c[t];
		}
	}
}

void HMM::BetaPass(vector<int> obsSeq)
{
	for (int i = 0; i < N; i++)
	{
		beta[T - 1][i] = c[T - 1];
	}

	for (int t = T - 2; t >= 0; t--)
	{
		for (int i = 0; i < N; i++)
		{
			beta[t][i] = 0;
			for (int j = 0; j < N; j++)
			{
				beta[t][i] += a[i][j] * b[j][obsSeq[t + 1]] * beta[t + 1][j];
			}

			//scale beta w/ same scale factor as alpha
			beta[t][i] *= c[t];
		}
	}
}

void HMM::GammaPass(vector<int> obsSeq)
{
	double denom;

	for (int t = 0; t < T - 1; t++)
	{
		denom = 0.0;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				denom += alpha[t][i] * a[i][j] * b[j][obsSeq[t + 1]] * beta[t + 1][j];
			}
		}
		for (int i = 0; i < N; i++)
		{
			gamma[t][i] = 0;
			for (int j = 0; j < N; j++)
			{
				digamma[t][i][j] = (alpha[t][i] * a[i][j] * b[j][obsSeq[t + 1]] * beta[t + 1][j]) / denom;
				gamma[t][i] += digamma[t][i][j];
			}
		}
	}

	//special case for gamma[T-1,i]
	denom = 0.0;
	for (int i = 0; i < N; i++)
	{
		denom += alpha[T - 1][i];
	}
	for (int i = 0; i < N; i++)
	{
		gamma[T - 1][i] = alpha[T - 1][i] / denom;
	}
}

void HMM::ReestimateModel(vector<int> obsSeq)
{
	//reestimate pi
	for (int i = 0; i < N; i++)
	{
		pi[i] = gamma[0][i];
	}

	//reestimate A
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			double numer = 0;
			double denom = 0;
			for (int t = 0; t < T - 1; t++)
			{
				numer += digamma[t][i][j];
				denom += gamma[t][i];
			}
			a[i][j] = numer / denom;
		}
	}
	

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			double numer = 0;
			double denom = 0;
			for (int t = 0; t < T; t++)
			{
				if (obsSeq[t] == j)
				{
					numer += gamma[t][i];
				}
				denom += gamma[t][i];
			}
			b[i][j] = numer / denom;
		}
	}
}

void HMM::ComputeLog(vector<int> obsSeq)
{
	logProb = 0.0;
	for (int i = 0; i < T; i++)
	{
		logProb += log(c[i]);
	}
	logProb = -logProb;
}

void HMM::PrintMatrices()
{
	//print pi matrix
	cout << "pi = { ";
	for (int i = 0; i < N; i++)
	{
		cout << pi[i] << " ";
	}
	cout << "}\n";

	cout << "a = {";
	for (int i = 0; i < N; i++)
	{
		cout << "{ ";
		for (int j = 0; j < N; j++)
		{
			//cout << std::setprecision(5) << a[i][j] << " ";
			printf("%.5f ", a[i][j]);
		}
		if (i < N - 1)
		{
			cout << "}\n     ";
		}
		else
		{
			cout << "}}\n";
		}
	}

	cout << "B Transpose = \n";
	for (int j = 0; j < M; j++)
	{
		cout << alphabet[j] + " ";

		for (int i = 0; i < N; i++)
		{
			//cout << std::setprecision(5) << b[i][j] << " ";
			printf("%.5f ", b[i][j]);
		}
		cout << "\n";
	}

	//cout << "Log Prob: " << logProb << "\n";
	printf("Log Prob: %f\n", logProb);
}

void HMM::RandInitMatrices()
{
	int i,
		j;

	double prob,
		ftemp,
		ftemp2;

	// initialize pseudo-random number generator
	srandom(seed);

	// initialize pi
	prob = 1.0 / (double)N;
	ftemp = prob / 10.0;
	ftemp2 = 0.0;
	for (i = 0; i < N; ++i)
	{
		if ((random() & 0x1) == 0)
		{
			pi[i] = prob + (double)(random() & 0x7) / 8.0 * ftemp;
		}
		else
		{
			pi[i] = prob - (double)(random() & 0x7) / 8.0 * ftemp;
		}
		ftemp2 += pi[i];

	}// next i

	for (i = 0; i < N; ++i)
	{
		pi[i] /= ftemp2;
	}

	// initialize A[][]
	prob = 1.0 / (double)N;
	ftemp = prob / 10.0;
	for (i = 0; i < N; ++i)
	{
		ftemp2 = 0.0;
		for (j = 0; j < N; ++j)
		{
			if ((random() & 0x1) == 0)
			{
				a[i][j] = prob + (double)(random() & 0x7) / 8.0 * ftemp;
			}
			else
			{
				a[i][j] = prob - (double)(random() & 0x7) / 8.0 * ftemp;
			}
			ftemp2 += a[i][j];

		}// next j

		for (j = 0; j < N; ++j)
		{
			a[i][j] /= ftemp2;
		}
	}

	// initialize B[][]
	prob = 1.0 / (double)M;
	ftemp = prob / 10.0;
	for (i = 0; i < N; ++i)
	{
		ftemp2 = 0.0;
		for (j = 0; j < M; ++j)
		{
			if ((random() & 0x1) == 0)
			{
				b[i][j] = prob + (double)(random() & 0x7) / 8.0 * ftemp;
			}
			else
			{
				b[i][j] = prob - (double)(random() & 0x7) / 8.0 * ftemp;
			}
			ftemp2 += b[i][j];

		}// next j

		for (j = 0; j < M; ++j)
		{
			b[i][j] /= ftemp2;
		}

	}// next i

}


void HMM::Run()
{
	while (iters < maxIters)
	{
		AlphaPass(obs);

		BetaPass(obs);

		GammaPass(obs);

		ReestimateModel(obs);

		ComputeLog(obs);

		iters++;

		double delta = abs(logProb - oldLogProb);
		if (iters < minIters || delta > epsilon)
		{
			oldLogProb = logProb;
			//goto step 3 (start of loop)
			//cout << iters << "\n";
		}
		else
		{
			break;
		}
	}
}

vector<HMM> hmms;
std::vector<int> obs;
vector<string> alphabet;
int numRestarts;
int N;

void* workerFunc(void* argv)
{
	int* args = (int*)argv;
	int start = args[0];
	int end = args[1];
	int index = args[2];

	//printf("thread %d: %d, %d\n", index, start, end);

	double highestLogProb = -DBL_MAX;
	for (int i = start; i < end; i++)
	{
		HMM cur;
		cur.init(obs, alphabet, N, i*(INT32_MAX / numRestarts));
		cur.RandInitMatrices();
		cur.Run();

		if (cur.logProb > highestLogProb)
		{
			highestLogProb = cur.logProb;
			hmms[index] = cur;
		}
	}

	return (void*)1;
}

std::string elapsedTime(std::chrono::system_clock::time_point start)
{
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	return std::to_string(elapsed.count()) + "ms";
}

//from stackoverflow lol https://stackoverflow.com/questions/236129/how-do-i-iterate-over-the-words-of-a-string
template<typename Out>
void split(const std::string &s, char delim, Out result) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}

int main(int argc, char* argv[])
{
	auto start = std::chrono::system_clock::now();

	string filename = argv[1];
	int maxWords = std::stoi(argv[2]);
	int numThreads = std::stoi(argv[3]);
	numRestarts = std::stoi(argv[4]);
	N = std::stoi(argv[5]);

	//read from the file
	std::ifstream t(filename);
	std::string data((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	t.close();

	//trim data to appropriate size
	vector<string> dataVec = split(data, ' ');
	if (dataVec.size() > maxWords)
	{
		dataVec.resize(maxWords);
	}

	std::set<string> alphabetSet(dataVec.begin(), dataVec.end());
	std::map<string, int> alphabetMap;
	int alphabetSize = 0;
	for (string s : alphabetSet) {
		alphabetMap[s] = alphabetSize;
		alphabet.push_back(s);
		cout << s << " ";
		alphabetSize++;
	}
	cout << "\n";

	if (N < 1)
	{
		N = alphabetSize;
	}

	//transform data into a form proper for a HMM
	for (int i = 0; i < dataVec.size(); i++)
	{
		obs.push_back(alphabetMap[dataVec[i]]);
		//cout << (int)obs[i] << ", ";
	}
	

	//start working
	printf("file: %s\nnum threads = %d, N = %d, random restarts = %d, T = %d, alphabet size = %d\n", filename.c_str(), numThreads, N, numRestarts, (int)obs.size(), alphabetSize);

	std::vector<pthread_t> rowWorkers(numThreads);
	std::vector<int> rowResult(numThreads);

	for (int i = 0; i < numThreads; i++)
	{
		hmms.push_back(HMM());
	}

	//create all the threads
	int args[numThreads][3];
	int curPos = 0;
	int threadsRemaining = numThreads;
	int restartsRemaining = numRestarts;
	for (int i = 0; i < numThreads; i++)
	{
		//int args[2] = { i * (numRestarts / numThreads), i * (numRestarts / numThreads) + (numRestarts / numThreads) };
		int g = (restartsRemaining / threadsRemaining);
		args[i][0] = curPos;
		curPos += g;
		args[i][1] = curPos;
		args[i][2] = i;

		pthread_create(&rowWorkers[i], NULL, workerFunc, (void*)args[i]);
		restartsRemaining -= g;
		threadsRemaining--;
	}
	printf("Created worker threads.\n");

	//join all of them
	for (int i = 0; i < numThreads; i++)
	{
		pthread_join(rowWorkers[i], (void**)&rowResult[i]);
	}

	//print thread status
	printf("Thread status: ");
	for (int i = 0; i < numThreads; i++)
	{
		printf("%d ", rowResult[i]);
	}
	printf("\n");

	//find the best scoring
	double highestLogProb = -DBL_MAX;
	int curHighest = 0;
	for (int i = 0; i < hmms.size(); i++)
	{

		if (hmms[i].logProb > highestLogProb)
		{
			highestLogProb = hmms[i].logProb;
			curHighest = i;
		}
	}

	cout << "-----FINAL-----\n";
	//Console.WriteLine("Key: " + key);
	hmms[curHighest].PrintMatrices();

	std::cout << "Execution took " << elapsedTime(start) << ".\n\n";

	return 0;
}
