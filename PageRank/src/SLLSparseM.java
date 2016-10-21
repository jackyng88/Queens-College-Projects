public class SLLSparseM implements SparseM {


    //Node class initialization
    private class Node {
        public int row;
        public int col;
        public int index;
        public int value;
        public Node next;

        // Node constructor


        private Node(int row, int col, int index, int value) {
            this.row = row;
            this.col = col;
            this.index = index;
            this.value = value;
            this.next = null;
        }

    }

    // SLLSparseM variables
    private int nrows;
    private int ncols;
    private Node head;
    private int nelements;




    // CONSTRUCTOR

    public SLLSparseM(int nrows, int ncols) {
        this.nrows = nrows;
        this.ncols = ncols;
        this.nelements = 0;
        this.head = null;
    }



    public boolean outOfBounds(int ridx, int cidx) {
        return ((ridx < 0) || (ridx >= nrows) || (cidx < 0) || (cidx >= ncols));
    }

    private int getIndex(int row, int col) {
        return (ncols * row) + col;
    }

    // Returns the number of rows
    public int nrows()
    {
        return nrows;
    }

    // Returns the number of columns
    public int ncols ()
    {
        return ncols;
    }

    // Returns the number of elements
    public int numElements ()
    {
        return nelements;
    }


    // Returns the element at a given entry
    public int getElement (int row, int col){
        if (outOfBounds(row, col))
            return 0;

        int index = getIndex(row, col);
        Node elem = head;

        while (elem != null) {
            if (elem.index != index) {
                elem = elem.next;
            } else {
                return elem.value;
            }
        }

        return 0;
    }



    public void clearElement (int row, int col){
        if (outOfBounds(row, col))
            return;

        int index = getIndex(row, col);

        if (head != null) {
            if (index == head.index) {
                head = head.next;
                nelements--;
                return;
            }
        }

        Node elem = head;
        Node prev = head;

        while (elem != null) {
            if (index > elem.index) {
                prev = elem;
                elem = elem.next;
            } else if (index == elem.index) {
                prev.next = elem.next;
                nelements--;
                return;
            } else {
                return;
            }
        }

    }

    // Erases/clears a node at specified row and column.
    public void setElement(int row,int col,int val){
        if (outOfBounds(row, col))
            return;

        int index = getIndex(row, col);


        if (val == 0) {
            clearElement(row, col);
        } else {
            Node elem = head;
            Node prev = head;

            if (head == null) {
                head = new Node(row, col, index, val);
                nelements++;
                return;
            } else if (index < head.index){
                Node temp = new Node(row, col, index, val);
                temp.next = head;
                head = temp;
                nelements++;
                return;
            }

            while (elem != null) {
                if (index > elem.index) {
                    prev = elem;
                    elem = elem.next;
                } else if (index == elem.index) {
                    elem.value = val;
                    return;
                } else {
                    Node temp = new Node(row, col, index, val);
                    temp.next = elem;
                    prev.next = temp;
                    nelements++;
                    return;
                }
            }

            if (elem == null) {
                prev.next = new Node(row, col, index, val);
                nelements++;
            }
        }
    }


    public void getAllElements (int[] ridx, int[] cidx, int[] val)
    {
        // What we have to do: insert values into these three arrays
        int counter = 0;

        Node elem = head;

        while (elem != null) {
            ridx[counter] = elem.row;
            cidx[counter] = elem.col;
            val[counter] = elem.value;
            counter++;
            elem = elem.next;
        }
    }


    public void addition (SparseM otherM){
        int nelem = otherM.numElements();
        int[] ridx = new int[nelem];
        int[] cidx = new int[nelem];
        int[] val = new int[nelem];
        otherM.getAllElements(ridx, cidx, val);
        int counter = 0;

        if (head == null) {
            int index = getIndex(ridx[counter], cidx[counter]);
            head = new Node(ridx[counter], cidx[counter], index, val[counter]);
            nelements++;
            counter++;
        }

        Node elem = head;
        Node prev = head;

        while (elem != null && counter < nelem) {
            int index = getIndex(ridx[counter], cidx[counter]);
            if (index > elem.index) {
                prev = elem;
                elem = elem.next;
            } else if (index == elem.index) {
                elem.value += val[counter];
                counter++;
                prev = elem;
                elem = elem.next;
            } else {
                setElement(ridx[counter], cidx[counter], val[counter]);
                counter++;
            }
        }

        if (elem == null) {
            while (counter < nelem) {
                int index = getIndex(ridx[counter], cidx[counter]);
                prev.next = new Node(ridx[counter], cidx[counter], index, val[counter]);
                prev = prev.next;
                nelements++;
                counter++;
            }
        }
    }
    // Adding two sparse matrices. .
}


