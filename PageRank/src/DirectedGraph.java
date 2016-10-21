
public class DirectedGraph {
    SLLSparseM sparseM;
    int numberOfNodes;

    // constructor initialize an undirected graph, n is the number of nodes
    public DirectedGraph(int n){
        sparseM = new SLLSparseM(n,n);
        numberOfNodes = n;
    }

    // check if the given node id is out of bounds
    private boolean outOfBounds(int nidx){
        return nidx >= numberOfNodes;
    }

    // set an edge (n1,n2)
    // beware of repeatingly setting a same edge and out-of-bound node ids
    public void setEdge(int n1, int n2){
        sparseM.setElement(n1,n2,1);
    }

    // compute page rank after num_iters iterations
    // then print them in a monotonically decreasing order
    void computePageRank(int num_iters){
        int[] ridx = new int[sparseM.numElements()];
        int[] cidx = new int[sparseM.numElements()];
        int[] val = new int[sparseM.numElements()];

        sparseM.getAllElements(ridx, cidx, val);

        int[] outDegrees = new int[numberOfNodes];

        for (int i = 0; i < ridx.length; i++)
        {
            outDegrees[ridx[i]] += 1;
        }

        double[] oldPageRank = new double[numberOfNodes];
        double[] newPageRank = new double[numberOfNodes];

        for (int i = 0; i < oldPageRank.length; i++) {
            oldPageRank[i] = 1;
        }

        // Go through each iteration
        for (int i = 1; i <= num_iters; i++){

            // Update page rank for every node
            for (int k = 0; k < numberOfNodes; k++){

                // Find each incoming edge
                for (int m = 0; m < cidx.length; m++) {

                    // ridx[m] gives us the STARTING NODE
                    // cidx[m] gives us the ENDING NODE
                    // k is which node we're updating page rank for
                    if (cidx[m] == k) {
                        newPageRank[k] +=  oldPageRank[ridx[m]] / outDegrees[ridx[m]];
                    }
                }
            }
            oldPageRank = newPageRank;
            newPageRank = new double[numberOfNodes];
        }

        LinkedList linkedList = new LinkedList();
        for(int i = 0 ; i < oldPageRank.length; i++) {
            linkedList.add(new Node(i, oldPageRank[i]));
        }
        linkedList.print();
    }
}
