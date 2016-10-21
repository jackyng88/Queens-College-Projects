public class LinkedList {
    private Node head;
    private int size;

    public LinkedList() {
        head = null;
        size = 0;
    }

    public void add(Node node) {
        if (head == null) {
            head = node;
            size++;
            return;
        } else if (node.pageRank > head.pageRank) {
            node.next = head;
            head = node;
            size++;
            return;
        } else if (size == 1) {
            head.next = node;
            size++;
            return;
        } else {
            Node elem = head.next;
            Node prev = head;

            while (elem != null) {
                if (node.pageRank > elem.pageRank) {
                    node.next = elem;
                    prev.next = node;
                    size++;
                    return;
                } else if (node.pageRank == elem.pageRank) {
                    node.next = elem;
                    prev.next = node;
                    size++;
                    return;
                } else {
                    prev = elem;
                    elem = elem.next;
                }
            }

            if (elem == null) {
                prev.next = node;
                size++;
            }
        }
    }

    public void print() {
        Node elem = head;
        while (elem != null) {
            System.out.println(elem.value + " " + elem.pageRank);
            elem = elem.next;
        }
    }
}