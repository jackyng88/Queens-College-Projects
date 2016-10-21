#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace std;

class EuclideanDistanceTransform
{
public:
    int numRows;
    int numCols;
    int minVal;
    int maxVal;
    int newMin;
    int newMax;
    double ** zeroFramedAry;
    int ** zeroFramedAryPP;
    double * neighborAry;
 
    EuclideanDistanceTransform (ifstream& infile);
    ~EuclideanDistanceTransform();
    void loadImage(ifstream& infile);
    void zeroFramed();
    void firstPassEuclidean(ofstream& out2);
    double firstPassNeighbors(int i, int j);
    void secondPassEuclidean(ofstream& out1, ofstream& out2);
    double secondPassNeighbors(int i, int j);
    void prettyPrintDistance(int ** arr, ofstream& out);
    void printArray(ofstream& out1);
  
};


EuclideanDistanceTransform::EuclideanDistanceTransform (ifstream& infile)
{
    //cout << "1" << endl;
    infile >> numRows;
    //cout << numRows;
    infile >> numCols;
    infile >> minVal;
    infile >> maxVal;
    
    //cout << numRows << " " << numCols << " " << minVal << " " << maxVal ;
    
    neighborAry = new double [5];
    
    //for the double zero framed array
    zeroFramedAry = new double * [numRows + 2];
    for (int i = 0; i < numRows + 2; i++)
    {
        zeroFramedAry[i] = new double[numCols + 2];
    }
    
    //for the integer zero framed array used for pretty print.
    zeroFramedAryPP = new int * [numRows + 2];
    for (int i = 0; i < numRows + 2; i++)
    {
        zeroFramedAryPP[i] = new int[numCols + 2];
    }
    
    zeroFramed();
    loadImage(infile);
    
}

EuclideanDistanceTransform::~EuclideanDistanceTransform(){}

void EuclideanDistanceTransform::zeroFramed()
{
    for(int i = 0; i < numRows + 2; i++)
    {
		zeroFramedAry[i][0] = 0;
		zeroFramedAry[i][numCols+1] = 0;
                
    }
	
    for(int j = 0; j < numCols + 2; j++)
    {
		zeroFramedAry[0][j] = 0;
		zeroFramedAry[numRows+1][j] = 0;	
    }
	
}

void EuclideanDistanceTransform::loadImage(ifstream& infile)
{
    //int ignoreheader1, ignoreheader2, ignoreheader3, ignoreheader4;
    //infile >> ignoreheader1 >> ignoreheader2 >> ignoreheader3 >> ignoreheader4;
                
    for (int i = 1; i < numRows + 1; i++)
    {
        for (int j = 1; j < numCols + 1; j++)
        {
            infile >> zeroFramedAry[i][j];
            //cout << zeroFramedAry[i][j] << " " ;
        }
    }
    
}

void EuclideanDistanceTransform::firstPassEuclidean(ofstream& out2)
{
        out2 << "------------------------------------------" << endl;
	out2 << " First Pass " << endl;
        
        for (int i = 0; i < numRows + 2; i++)
        {
            for (int j = 0; j < numCols + 2; j++)
            {
                if (zeroFramedAry[i][j] > 0)
                {
                    zeroFramedAry[i][j] = firstPassNeighbors (i,j);
                    zeroFramedAryPP[i][j] = firstPassNeighbors (i,j) + 0.5;
                }
            }
        }
        
        out2 << "Debugging Pretty Print of Pass 1 " << endl;
        prettyPrintDistance(zeroFramedAryPP,out2);
        
}

double EuclideanDistanceTransform::firstPassNeighbors(int i, int j)
{
        //Initialize neighbor array to all 0's
        //neighborAry = {0.0, 0.0, 0.0, 0.0, 0.0};
    
        for (int k = 0; k < 5; k++)
        {
            neighborAry[k] = 0;
        }
        
 
	neighborAry[0] = zeroFramedAry[i-1][j] + 1;
        neighborAry[1] = zeroFramedAry[i][j-1] + 1;
	neighborAry[2] = zeroFramedAry[i-1][j+1] + sqrt(2);
        neighborAry[3] = zeroFramedAry[i-1][j-1] + sqrt(2);
        
        //sort
        for(int x = 0; x < 4; x++)
        {
            for(int y = x + 1; y < 4; y++)
            {				
		if(neighborAry[y] < neighborAry[x])
                {
                    int t = neighborAry[x];
                    neighborAry[x] = neighborAry[y];
                    neighborAry[y] = t;
		}
            }
        }
    
        return neighborAry[0];
}


void EuclideanDistanceTransform::secondPassEuclidean(ofstream& out1, ofstream& out2)
{
        //out1 << numRows << " " << numCols << " " << minVal << " " << maxVal ;
    
        newMin = 0;
        newMax = 0;
        
        out2 << "------------------------------------------" << endl;
	out2 << " Second Pass " << endl;
        
        for (int i = numRows + 1; i > 0; i--)
        {
            for (int j = numCols + 1; j > 0; j--)
            {
                if (zeroFramedAry[i][j] > 0)
                {
                    zeroFramedAry[i][j] = secondPassNeighbors (i,j);
                    zeroFramedAryPP[i][j] = secondPassNeighbors (i,j) + 0.5;
                    
                    //Computing newMin and newMax for the printing of the 2D Double Array. Ceiling function
                    //on value to create integer.
                    if (zeroFramedAryPP[i][j] < newMin)
                    {
                        newMin = zeroFramedAryPP[i][j];
                    }
                    if (zeroFramedAryPP[i][j] > newMax)
                    {
                        newMax = zeroFramedAryPP[i][j];
                    }
                }
            }
        }
        
        out2 << "Debugging Pretty Print of Pass 2 " << endl;
        prettyPrintDistance(zeroFramedAryPP,out2);
        printArray(out1);
    
}

double EuclideanDistanceTransform::secondPassNeighbors(int i, int j)
{
    //Initialize neighbor array to all 0's. Especially for second pass.
  
        //neighborAry = {0.0, 0.0, 0.0, 0.0, 0.0};
        for (int k = 0; k < 5; k++)
        {
            neighborAry[k] = 0;
        }
        
        neighborAry[0] = zeroFramedAry[i][j];
	neighborAry[1] = zeroFramedAry[i+1][j] + 1;
        neighborAry[2] = zeroFramedAry[i][j+1] + 1;
	neighborAry[3] = zeroFramedAry[i-1][j-1] + sqrt(2);
        //Second pass has the pixel itself included as neighbor.
        neighborAry[4] = zeroFramedAry[i+1][j+1] + sqrt(2);
        
        //sorting
        for(int x = 0; x < 5; x++)
        {
            for(int y = x + 1; y < 5; y++)
            {				
		if(neighborAry[y] < neighborAry[x])
                {
                    int t = neighborAry[x];
                    neighborAry[x] = neighborAry[y];
                    neighborAry[y] = t;
		}
            }
        }
        
        return neighborAry[0];
}

void EuclideanDistanceTransform::prettyPrintDistance(int** arr, ofstream& out)
{
    for(int i = 1; i < numRows + 1; i++)
    {
	for(int j = 1; j < numCols + 1; j++)
        {
            if(arr[i][j] == 0)
            {
		out << " " << setw(2);
            }
				
            else
            {
                out << arr[i][j] << setw(2);
            }
			
        }
        out << endl;
    }
}

void EuclideanDistanceTransform::printArray (ofstream& out1)
{
    out1 << numRows << " " << numCols << " " << newMin << " " << newMax << endl;
    for (int i = 1; i < numRows + 1; i++)
    {
        for (int j = 1; j < numCols + 1; j++)
        {
            out1 << zeroFramedAry[i][j] << " ";
        }
    }
}


int main(int argc, char *argv[]) 
{

    ifstream infile;
    ofstream out1;
    ofstream out2;
    
    infile.open (argv[1]);
    out1.open (argv[2]);
    out2.open (argv[3]);
    
    //cout << "0" << endl;
    EuclideanDistanceTransform newEDT (infile);
    //cout << "1" << endl;
    //infile.close();
    //infile.open(argv[1]);
    
    //newEDT.loadImage(infile);
    newEDT.firstPassEuclidean(out2);
    //cout << "2" << endl;
    newEDT.secondPassEuclidean(out1, out2);
    
    return 0;
}

