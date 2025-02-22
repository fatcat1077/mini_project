#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node {
    int key;
    int degree;
    struct Node *parent;
    struct Node *child;
    struct Node *left;
    struct Node *right;
    int marked;
} Node;

typedef struct FibonacciHeap {
    Node *min;
    int size;
} FibonacciHeap;

// Create new node
Node* createNode(int key) {
    Node *node = (Node*)malloc(sizeof(Node));
    node->key = key;
    node->degree = 0;
    node->parent = NULL;
    node->child = NULL;
    node->left = node;
    node->right = node;
    node->marked = 0;
    return node;
}

// Create new heap
FibonacciHeap* createHeap() {
    FibonacciHeap *heap = (FibonacciHeap*)malloc(sizeof(FibonacciHeap));
    heap->min = NULL;
    heap->size = 0;
    return heap;
}

// Insert node into root list
void insertRootList(FibonacciHeap *heap, Node *node) {
    if (heap->min == NULL) {
        heap->min = node;
    } else {
        node->right = heap->min->right;
        node->left = heap->min;
        heap->min->right->left = node;
        heap->min->right = node;
        if (node->key < heap->min->key) {
            heap->min = node;
        }
    }
    heap->size++;
}

// Insert operation
void insert(FibonacciHeap *heap, int key) {
    Node *node = createNode(key);
    insertRootList(heap, node);
}

// Link two nodes
void link(FibonacciHeap *heap, Node *y, Node *x) {
    // Remove y from root list
    y->left->right = y->right;
    y->right->left = y->left;
    
    // Make y a child of x
    y->parent = x;
    if (x->child == NULL) {
        x->child = y;
        y->right = y;
        y->left = y;
    } else {
        y->right = x->child->right;
        y->left = x->child;
        x->child->right->left = y;
        x->child->right = y;
    }
    x->degree++;
    y->marked = 0;
}

// Consolidate heap
void consolidate(FibonacciHeap *heap) {
    int maxDegree = heap->size + 1;
    Node *A[100] = {NULL};  // Array for consolidation
    
    Node *start = heap->min;
    Node *current = start;
    if (current == NULL) return;
    
    // Count roots and process them
    do {
        Node *x = current;
        current = current->right;
        int d = x->degree;
        
        while (A[d] != NULL) {
            Node *y = A[d];
            if (x->key > y->key) {
                Node *temp = x;
                x = y;
                y = temp;
            }
            link(heap, y, x);
            A[d] = NULL;
            d++;
        }
        A[d] = x;
    } while (current != start);
    
    // Rebuild root list
    heap->min = NULL;
    for (int i = 0; i < maxDegree; i++) {
        if (A[i] != NULL) {
            if (heap->min == NULL) {
                heap->min = A[i];
                A[i]->left = A[i];
                A[i]->right = A[i];
            } else {
                insertRootList(heap, A[i]);
            }
        }
    }
}

// Extract minimum
Node* extractMin(FibonacciHeap *heap) {
    Node *z = heap->min;
    if (z != NULL) {
        // Add children to root list
        if (z->child != NULL) {
            Node *child = z->child;
            do {
                Node *next = child->right;
                insertRootList(heap, child);
                child->parent = NULL;
                child = next;
            } while (child != z->child);
        }
        
        // Remove z from root list
        z->left->right = z->right;
        z->right->left = z->left;
        
        if (z == z->right) {
            heap->min = NULL;
        } else {
            heap->min = z->right;
            consolidate(heap);
        }
        heap->size--;
    }
    return z;
}

// Cut node from parent
void cut(FibonacciHeap *heap, Node *x, Node *y) {
    if (x == x->right) {
        y->child = NULL;
    } else {
        x->left->right = x->right;
        x->right->left = x->left;
        if (y->child == x) {
            y->child = x->right;
        }
    }
    y->degree--;
    insertRootList(heap, x);
    x->parent = NULL;
    x->marked = 0;
}

// Cascading cut
void cascadingCut(FibonacciHeap *heap, Node *y) {
    Node *z = y->parent;
    if (z != NULL) {
        if (!y->marked) {
            y->marked = 1;
        } else {
            cut(heap, y, z);
            cascadingCut(heap, z);
        }
    }
}

// Decrease key
void decreaseKey(FibonacciHeap *heap, Node *x, int value) {
    x->key -= value;
    Node *y = x->parent;
    if (y != NULL && x->key < y->key) {
        cut(heap, x, y);
        cascadingCut(heap, y);
    }
    if (x->key < heap->min->key) {
        heap->min = x;
    }
}

// Find node with given key
Node* findNode(FibonacciHeap *heap, int key) {
    Node *current = heap->min;
    if (current == NULL) return NULL;
    
    do {
        if (current->key == key) return current;
        Node *child = current->child;
        if (child != NULL) {
            Node *found = findNode((FibonacciHeap*)child, key);
            if (found != NULL) return found;
        }
        current = current->right;
    } while (current != heap->min);
    return NULL;
}

// Delete operation
void delete(FibonacciHeap *heap, int key) {
    Node *x = findNode(heap, key);
    if (x != NULL) {
        decreaseKey(heap, x, x->key + 1);  // Decrease to ensure it's minimum
        extractMin(heap);
    }
}

// Print level order for a single tree
void printLevelOrder(Node *root) {
    if (root == NULL) return;
    
    Node *queue[100];
    int front = 0, rear = 0;
    queue[rear++] = root;
    
    int currentLevel = 0;
    int nodesAtLevel = 1;
    int nextLevelNodes = 0;
    
    while (front < rear) {
        Node *node = queue[front++];
        printf("%d", node->key);
        nodesAtLevel--;
        
        Node *child = node->child;
        if (child != NULL) {
            do {
                queue[rear++] = child;
                nextLevelNodes++;
                child = child->right;
            } while (child != node->child);
        }
        
        if (nodesAtLevel == 0) {
            if (front < rear) printf(" ");
            nodesAtLevel = nextLevelNodes;
            nextLevelNodes = 0;
            currentLevel++;
        } else if (front < rear) {
            printf(" ");
        }
    }
}

// Print heap structure
void printHeap(FibonacciHeap *heap) {
    if (heap->min == NULL) return;
    
    // Store roots by degree
    Node *trees[100] = {NULL};
    int degrees[100] = {0};
    int count = 0;
    
    Node *current = heap->min;
    do {
        trees[count] = current;
        degrees[count] = current->degree;
        count++;
        current = current->right;
    } while (current != heap->min);
    
    // Sort by degree
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (degrees[j] > degrees[j + 1]) {
                Node *temp = trees[j];
                trees[j] = trees[j + 1];
                trees[j + 1] = temp;
                int tempDeg = degrees[j];
                degrees[j] = degrees[j + 1];
                degrees[j + 1] = tempDeg;
            }
        }
    }
    
    // Print each tree
    for (int i = 0; i < count; i++) {
        printLevelOrder(trees[i]);
        if (i < count - 1) printf("\n");
    }
}

int main() {
    FibonacciHeap *heap = createHeap();
    char command[20];
    int key, value;
    
    while (1) {
        scanf("%s", command);
        if (strcmp(command, "exit") == 0) break;
        
        if (strcmp(command, "insert") == 0) {
            scanf("%d", &key);
            insert(heap, key);
        }
        else if (strcmp(command, "delete") == 0) {
            scanf("%d", &key);
            delete(heap, key);
        }
        else if (strcmp(command, "decrease") == 0) {
            scanf("%d %d", &key, &value);
            Node *node = findNode(heap, key);
            if (node != NULL) decreaseKey(heap, node, value);
        }
        else if (strcmp(command, "extract-min") == 0) {
            Node *min = extractMin(heap);
            if (min != NULL) free(min);
        }
    }
    
    printHeap(heap);
    return 0;
}