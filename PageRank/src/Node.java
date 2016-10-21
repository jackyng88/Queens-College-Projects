public class Node {
    public int value;
    public double pageRank;
    public Node next;

    public Node(int val, double rank) {
        value = val;
        pageRank = rank;
    }
}