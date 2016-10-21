import java.util.ArrayList;


public class UndirectedGraph {

    private ArrayList<Node> nodes = new ArrayList<>();
    private	int size = 0;

    // constructor initialize an undirected graph, n is the number of nodes
    public UndirectedGraph(int n) {

        size = n;
        for (int i = 0; i < n; i++) {
            nodes.add(new Node(i));


        }
    }

    // check if the given node id is out of bounds
    private boolean outOfBounds(int nidx){
        return nidx > size;
    }

    // set an edge (n1,n2).
    // Since this is an undirected graph, (n2,n1) is also set to one
    public void setEdge(int n1, int n2){

        if (!outOfBounds(n1) && !outOfBounds(n2)){
            Node firstNode = nodes.get(n1);
            Node secondNode = nodes.get(n2);
            firstNode.addIncomingEdge(secondNode);
            firstNode.addOutgoingEdge(secondNode);
            secondNode.addIncomingEdge(firstNode);
            secondNode.addOutgoingEdge(secondNode);
        }

    }

    // print an output soft clique in one line
    public void printClique(ArrayList<Integer> nlist){
        for(int i = 0; i < nlist.size(); ++i)
            System.out.print(nlist.get(i) + " ");
        System.out.println("");
    }

    // compute maximal soft clique
    // cliquesize_lower_bd: k
    // num_missing_edges: l
    public void findMaxSoftClique(int cliquesize_lower_bd, int num_missing_edges){

    }

    // compute maximal soft clique by using recursion
    // to compute all (k,l) soft cliques using recursion
    // you should check the partial subset during generation
    // rather than checking the whole subset
    // cliquesize_lower_bd: k
    // num_missing_edges: l
    public void findMaxSoftCliqueAdvanced(int cliquesize_lower_bd, int num_missing_edges){

    }

}
