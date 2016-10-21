#include <iostream>
#include <fstream>
#include <string>
using namespace std;


class HeapSort
{
public:
    int* heapAry;
    int count;
    int left;
    int right;
    int root;
    int parent;
    int index;
    int tempindex;
    
    //Constructor
    HeapSort(int count)
    {
        heapAry = new int [count+1];
        heapAry[0] = 0;
        index = 1;
        left = NULL;
        right = NULL;
        parent = NULL;
        tempindex = NULL;
    }
    
    //Destructor
    ~HeapSort()
    {
        delete[] heapAry;
    }
    
    void buildHeap(ifstream& infile)
    {
        int data;
        while (infile >> data)
        {  
        insertOneDataItem(data);
        }
    }
    
    bool isLeaf()
    {
     return (left > heapAry[0] && right > heapAry[0]);  
    }
    
    bool isRoot()
    {
     return (index == 1);  
    }
    
    void deleteHeap(ifstream& infile, ofstream& out)
    {
        int data;

        while (infile >> data)
        {
            deleteRoot(data,out);
        }        
    }
    
    void insertOneDataItem(int data)
    {
        if (isHeapFull())
        {
            return;
        }        
        heapAry[index] = data;
        cout << heapAry[index] << endl;
        tempindex = index;
        bubbleUp(heapAry[index]);
        index++;
        heapAry[0]++;
    }
    
    
    void deleteRoot(int data, ofstream& out)
    {
       int temp;
       if (isHeapEmpty())
       {
           return;
       }
       
       out << heapAry[1] <<endl;
       cout << "Root: " << heapAry[1] << endl;
       temp = heapAry[0];
       cout << "Placing element: " << temp << " with value: " << heapAry[temp] << " as the root." << endl;
       heapAry[1] = heapAry[temp];
       heapAry[temp] = NULL;
       temp--;
       heapAry[0] = temp;
       tempindex = 1;
       bubbleDown(heapAry[tempindex]); 
    }
    
    
    void bubbleUp(int data)
    {
        //Checks to see if value of node is greater than parent or not.
        //If it is indeed smaller, swap places with the parent. Index is count. 
        //If count = 30, index = 30. (ie) heapAry[30] = 90.

        if (tempindex == 1)
        {
            return;
        }
        parent = tempindex/2;
        //Check to see if parent of this node (index/2) has data larger than index.
        //If yes, swap. Index swaps as well, and we continue bubbling up.
        if (heapAry[parent] > heapAry[tempindex])
        {
            int temp;
            temp = heapAry[tempindex];
            heapAry[tempindex] = heapAry[parent];
            heapAry[parent] = temp;
            //index = parent;
            //bubbleUp(heapAry[index]);
            tempindex = parent;
            bubbleUp(heapAry[tempindex]);

        }
        
        //After swapping, tempindex gets to a point where it's stuck. While loop
        //continues to jump through parents. 
        while (tempindex != 1)
        {
            tempindex = tempindex/2; 
        }

        bubbleUp(heapAry[tempindex]);
    }
    
    void bubbleDown(int data)
    {
        //Needs to check which is the smaller of two between left and right child.
        //Then after checking compares the parent with that child.
        //If larger, bubble down.
        left = (tempindex * 2);
        right = (tempindex * 2) + 1;
        int leafparent;
   
        //Checks to see if the current tempindex is a leaf node or not.
        //If yes, return.
        if (left > heapAry[0] || right > heapAry[0])
        {
            return;
        }
        
        if (heapAry[left] <= heapAry[right])
        {
            if (heapAry[tempindex] > heapAry[left] && left <= heapAry[0])
            {
                //swap
                int temp;
                temp = heapAry[left];
                cout << "Swapping index: "<< left << " with value: " << heapAry[left] << " with " << heapAry[tempindex] << " at index: " << tempindex << endl;
                heapAry[left] = heapAry[tempindex];
                heapAry[tempindex] = temp;
                tempindex = left;
                
                //If the bubbling down reaches a leaf node. Still need to set tempindex to become the parent 
                //to check to see if another swap is needed between parent and the other leafnode.
                left = (tempindex *2);
                right = (tempindex *2) + 1;
                
                if (left > heapAry[0] || right > heapAry[0])
                {
                     //leafparent = (tempindex/2);
                    tempindex = tempindex/2;
                     bubbleDown(tempindex);
                }
                
                bubbleDown(heapAry[tempindex]);
            }
        }

        if (heapAry[right] < heapAry[left])
        {
            if (heapAry[tempindex] > heapAry[right] && right <= heapAry[0])
            {
                int temp;
                temp = heapAry[right];
                cout << "Swapping index: " << right << " with value: " << heapAry[right] << " with " << heapAry[tempindex] << " at index: " << tempindex << endl;

                heapAry[right] = heapAry[tempindex];
                heapAry[tempindex] = temp;
                tempindex = right;
                
                left = (tempindex *2);
                right = (tempindex *2) + 1;
                
                if (left > heapAry[0] || right > heapAry[0])
                {
                     //leafparent = (tempindex/2);
                    tempindex = tempindex/2;
                     bubbleDown(tempindex);
                }
                
                bubbleDown(heapAry[tempindex]);

            }   
          
        }

    }
    
    bool isHeapEmpty()
    {
        if (heapAry[0] == 0)
        {
            return true;
        }
        
        else
        {
            return false;
        }
    }
    
    bool isHeapFull()
    {
        return (index == count);   
    }
    
    void printHeap(ofstream& out)
    {
        for (int i = 1; i < heapAry[0]; ++i)
        {
            out << heapAry[i] <<endl;

        } 
    }
};



/*
 * 
 */
int main(int argc, char *argv[]) {

    int data;
    ifstream infile;
    ofstream out1;
    ofstream out2;
    int count = 0;

    //Reading from file.
    infile.open(argv[1]);
    cout << "Reading Data from the File. " << endl;
    while (infile >> data)
    {
        cout << data << endl;
        count++;  
    }
    infile.close();
 
    //Instantiating the tree.
    HeapSort myHeap(count);

    //BuildHeap. Move these into the buildHeap function?
    cout << "Populating the tree with data from the file." << endl;
    cout << "There are: " << count <<" integers in the file." << endl;
    
    //Reading from file and populating the array with values from file.
    infile.open(argv[1]);
    myHeap.buildHeap(infile);
    out1.open(argv[2]);
    myHeap.printHeap(out1);
    infile.close();
    
    
    //Reading from file to delete.
    //Delete Heap. Move these into the deleteHeap function?
    //myHeap.deleteHeap();)
    infile.open(argv[2]);
    out2.open (argv[3]);
    cout << "Deleting the tree and outputting the sorted tree to a separate file." << endl;
    myHeap.deleteHeap(infile,out2);
    infile.close();
    out1.close();
    out2.close();
    
    

    return 0;
}

