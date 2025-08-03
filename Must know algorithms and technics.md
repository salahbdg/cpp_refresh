# Complete Competitive Programming Algorithms Guide

## Part 1: Algorithm Explanations 

### What is a Graph?
Think of a graph like a map of cities (nodes/vertices) connected by roads (edges). You can represent friendships on Facebook, roads between cities, or dependencies between tasks.

### What is Dynamic Programming?
**Core idea:** Break a big problem into smaller subproblems, solve each once, and reuse the answers.
**Real-world analogy:** Like taking notes in class - instead of re-deriving the same formula every time, you write it down once and refer back to it.

---

## Part 2: Graph Algorithms

### DFS (Depth-First Search)
**What it does:** Explores a graph by going as deep as possible before backtracking.

**Real-world analogy:** Like exploring a maze - you keep going forward until you hit a dead end, then you backtrack and try a different path.

**When to use:**
- Finding connected components (groups of friends)
- Detecting cycles (circular dependencies)  
- Topological sorting (ordering tasks with dependencies)

**How it works:**
1. Start at a node, mark it as visited
2. Go to an unvisited neighbor and repeat
3. If no unvisited neighbors, backtrack
4. Continue until all reachable nodes are visited

#### Graph Representation
```cpp
// Adjacency List (most common)
vector<vector<int>> adj(n);  // for n vertices
adj[u].push_back(v);  // add edge u -> v

// Weighted graph
vector<vector<pair<int, int>>> adj(n);  // {destination, weight}
adj[u].push_back({v, weight});

// Adjacency Matrix (for dense graphs)
vector<vector<int>> adj(n, vector<int>(n, 0));
adj[u][v] = 1;  // or weight
```

#### DFS Implementation
```cpp
vector<bool> visited(n, false);

void dfs(int node, vector<vector<int>>& adj) {
    visited[node] = true;
    // Process node here
    
    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            dfs(neighbor, adj);
        }
    }
}

// Check if graph is connected
bool isConnected() {
    dfs(0, adj);
    for (bool v : visited) {
        if (!v) return false;
    }
    return true;
}
```

### BFS (Breadth-First Search)
**What it does:** Explores a graph level by level, visiting all neighbors before going deeper.

**Real-world analogy:** Like ripples in a pond - you explore all nodes 1 step away, then 2 steps away, then 3 steps away, etc.

**When to use:**
- Finding shortest path in unweighted graphs
- Level-order traversal
- Finding minimum number of steps/moves

**How it works:**
1. Start at a node, add it to a queue
2. Remove node from queue, visit all its unvisited neighbors
3. Add those neighbors to the queue
4. Repeat until queue is empty

#### BFS Implementation
```cpp
vector<int> bfs(int start, vector<vector<int>>& adj) {
    vector<bool> visited(adj.size(), false);
    vector<int> distance(adj.size(), -1);
    queue<int> q;
    
    q.push(start);
    visited[start] = true;
    distance[start] = 0;
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                distance[neighbor] = distance[node] + 1;
                q.push(neighbor);
            }
        }
    }
    return distance;
}
```

### Dijkstra's Algorithm
**What it does:** Finds the shortest path in a weighted graph (roads with different distances).

**Real-world analogy:** Like GPS navigation - finding the fastest route considering traffic, distance, etc.

**When to use:**
- Shortest path with positive weights
- Network routing
- Flight connections with costs

**How it works:**
1. Set distance to start = 0, all others = infinity
2. Always process the closest unvisited node
3. Update distances to its neighbors if we found a shorter path
4. Repeat until all nodes processed

#### Dijkstra Implementation
```cpp
vector<int> dijkstra(int start, vector<vector<pair<int, int>>>& adj) {
    int n = adj.size();
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    
    dist[start] = 0;
    pq.push({0, start});  // {distance, node}
    
    while (!pq.empty()) {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        
        if (d > dist[u]) continue;
        
        for (auto& edge : adj[u]) {
            int v = edge.first;
            int weight = edge.second;
            
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}
```

### Floyd-Warshall
**What it does:** Finds shortest paths between ALL pairs of nodes.

**Real-world analogy:** Creating a complete distance table between all cities, considering all possible routes.

**When to use:**
- All-pairs shortest paths
- Small graphs (≤ 400 nodes)
- Transitive closure problems

**How it works:**
1. Try using each node as an intermediate point
2. If going through node k makes path from i to j shorter, update it
3. Do this for all possible intermediate nodes

#### Floyd-Warshall Implementation
```cpp
void floydWarshall(vector<vector<int>>& dist) {
    int n = dist.size();
    
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
}
```

### Topological Sort
**What it does:** Orders nodes so that for every directed edge A→B, A comes before B in the ordering.

**Real-world analogy:** Scheduling tasks with prerequisites (you must take Math 101 before Math 201).

**When to use:**
- Task scheduling with dependencies
- Course prerequisites
- Build systems (compile dependencies)

**How it works:**
1. Find nodes with no incoming edges (no prerequisites)
2. Remove them and add to result
3. Remove their outgoing edges
4. Repeat until all nodes processed

#### Topological Sort Implementation
```cpp
vector<int> topologicalSort(vector<vector<int>>& adj) {
    int n = adj.size();
    vector<int> indegree(n, 0);
    
    // Calculate indegrees
    for (int i = 0; i < n; i++) {
        for (int neighbor : adj[i]) {
            indegree[neighbor]++;
        }
    }
    
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }
    
    vector<int> result;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);
        
        for (int neighbor : adj[node]) {
            indegree[neighbor]--;
            if (indegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }
    
    return result;  // Empty if cycle exists
}
```

### Union-Find (Disjoint Set Union)
**What it does:** Efficiently tracks which nodes belong to the same connected group.

**Real-world analogy:** Managing friend groups - quickly check if two people are in the same social circle, or merge two friend groups.

**When to use:**
- Dynamic connectivity queries
- Kruskal's MST algorithm
- Detecting cycles in undirected graphs

**How it works:**
1. Each node starts as its own group
2. **Union:** Merge two groups together
3. **Find:** Check which group a node belongs to (with path compression)

#### Union-Find Implementation
```cpp
class UnionFind {
private:
    vector<int> parent, rank;
    
public:
    UnionFind(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // path compression
        }
        return parent[x];
    }
    
    void unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;
        
        // Union by rank
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
    }
    
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};
```

---

## Part 3: Dynamic Programming Patterns

### Fibonacci / Linear DP
**Problem:** Find the nth Fibonacci number (1, 1, 2, 3, 5, 8, 13...)

**Naive approach:** Recursively calculate fib(n-1) + fib(n-2) - but this recalculates the same values many times!

**DP approach:** Calculate from bottom up, storing each result.

**Why DP helps:** Without DP: exponential time. With DP: linear time.

#### Implementation
```cpp
// Bottom-up
vector<int> dp(n + 1);
dp[0] = dp[1] = 1;
for (int i = 2; i <= n; i++) {
    dp[i] = dp[i-1] + dp[i-2];
}

// Space optimized
int prev2 = 1, prev1 = 1;
for (int i = 2; i <= n; i++) {
    int curr = prev1 + prev2;
    prev2 = prev1;
    prev1 = curr;
}
```

### Longest Common Subsequence (LCS)
**Problem:** Find the longest sequence that appears in both strings (doesn't need to be consecutive).

**Example:** 
- String 1: "ABCDGH"
- String 2: "AEDFHR"  
- LCS: "ADH" (length 3)

**Real-world use:** DNA sequence comparison, diff tools in programming, plagiarism detection.

**How it works:**
1. If characters match, LCS = 1 + LCS of remaining strings
2. If they don't match, LCS = max(LCS skipping char from string1, LCS skipping char from string2)

#### LCS Implementation
```cpp
int lcs(string& s1, string& s2) {
    int m = s1.length(), n = s2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}
```

### Knapsack Problem
**Problem:** You have a backpack with weight limit W. You have items with weights and values. Maximize value without exceeding weight.

**Real-world analogy:** Packing for vacation with luggage weight limits, or choosing which stocks to buy with limited money.

**How it works:**
1. For each item, decide: include it or skip it
2. If including: value = item_value + best_value_with_remaining_capacity
3. If skipping: value = best_value_without_this_item
4. Choose the better option

#### Knapsack Implementation
```cpp
// 0/1 Knapsack
int knapsack(vector<int>& weights, vector<int>& values, int W) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 1; w <= W; w++) {
            if (weights[i-1] <= w) {
                dp[i][w] = max(dp[i-1][w], 
                              dp[i-1][w - weights[i-1]] + values[i-1]);
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    return dp[n][W];
}
```

### Longest Increasing Subsequence (LIS)
**Problem:** Find the longest subsequence where elements are in increasing order.

**Example:** [3, 1, 4, 1, 5, 9, 2, 6] → LIS is [1, 4, 5, 9] or [1, 4, 5, 6] (length 4)

**Real-world use:** Stock prices (longest period of growth), patience sorting.

**Two approaches:**
1. **O(n²):** For each element, find longest sequence ending at that element
2. **O(n log n):** Maintain array of smallest tail elements for each length

#### LIS Implementation
```cpp
// O(n^2) version
int lis(vector<int>& arr) {
    int n = arr.size();
    vector<int> dp(n, 1);
    
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }
    return *max_element(dp.begin(), dp.end());
}

// O(n log n) version
int lisOptimal(vector<int>& arr) {
    vector<int> tails;
    for (int num : arr) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }
    return tails.size();
}
```

---

## Part 4: String Algorithms

### KMP (Knuth-Morris-Pratt)
**Problem:** Find all occurrences of pattern in text efficiently.

**Naive approach:** Check every position in text - O(nm) time where n=text length, m=pattern length.

**KMP improvement:** When mismatch occurs, don't start over completely - use information about pattern to skip characters.

**Real-world analogy:** Like proofreading - if you're looking for "ababac" and find "ababa" then 'x', you don't need to start over from the beginning; you know the next comparison should start from "aba".

**Key insight:** Pre-compute "failure function" that tells you how much of the pattern you can skip when a mismatch occurs.

#### KMP Implementation
```cpp
vector<int> computeLPS(string& pattern) {
    int m = pattern.length();
    vector<int> lps(m, 0);
    int len = 0, i = 1;
    
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}

vector<int> KMP(string& text, string& pattern) {
    vector<int> lps = computeLPS(pattern);
    vector<int> result;
    int i = 0, j = 0;
    int n = text.length(), m = pattern.length();
    
    while (i < n) {
        if (pattern[j] == text[i]) {
            i++; j++;
        }
        
        if (j == m) {
            result.push_back(i - j);
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    return result;
}
```

### Manacher's Algorithm
**Problem:** Find the longest palindrome in a string.

**Naive approach:** Check every possible center - O(n²) time.

**Manacher's improvement:** Use information about previously found palindromes to avoid redundant work - O(n) time.

**Key insight:** If you're inside a larger palindrome, you can use symmetry to get information about the current position.

#### Manacher Implementation
```cpp
string preprocess(string s) {
    string result = "^";
    for (char c : s) {
        result += "#" + string(1, c);
    }
    result += "#$";
    return result;
}

string longestPalindrome(string s) {
    string T = preprocess(s);
    int n = T.length();
    vector<int> P(n, 0);
    int C = 0, R = 0;
    
    for (int i = 1; i < n - 1; i++) {
        int mirror = 2 * C - i;
        
        if (i < R) {
            P[i] = min(R - i, P[mirror]);
        }
        
        while (T[i + (1 + P[i])] == T[i - (1 + P[i])]) {
            P[i]++;
        }
        
        if (i + P[i] > R) {
            C = i;
            R = i + P[i];
        }
    }
    
    int maxLen = 0, centerIndex = 0;
    for (int i = 1; i < n - 1; i++) {
        if (P[i] > maxLen) {
            maxLen = P[i];
            centerIndex = i;
        }
    }
    
    int start = (centerIndex - maxLen) / 2;
    return s.substr(start, maxLen);
}
```

---

## Part 5: Mathematical Algorithms

### GCD (Greatest Common Divisor)
**Problem:** Find the largest number that divides both a and b.

**Example:** GCD(12, 18) = 6

**Euclidean Algorithm:** 
- GCD(a, b) = GCD(b, a mod b)
- Keep reducing until one number becomes 0

**Why it works:** The GCD doesn't change when you replace the larger number with the remainder.

#### GCD and LCM Implementation
```cpp
int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}

int lcm(int a, int b) {
    return (a / gcd(a, b)) * b;
}

// Extended Euclidean Algorithm
int extgcd(int a, int b, int& x, int& y) {
    if (b == 0) {
        x = 1; y = 0;
        return a;
    }
    int x1, y1;
    int d = extgcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return d;
}
```

### Sieve of Eratosthenes
**Problem:** Find all prime numbers up to n.

**Real-world analogy:** Like crossing out multiples on a hundreds chart.

**How it works:**
1. Start with all numbers marked as "potentially prime"
2. Start with 2: mark all its multiples (4, 6, 8, ...) as composite
3. Move to next unmarked number (3): mark all its multiples as composite
4. Continue until you've processed all numbers up to √n

**Why √n:** If a number n has a factor larger than √n, it must also have a factor smaller than √n.

#### Sieve Implementation
```cpp
vector<bool> sieve(int n) {
    vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    
    for (int i = 2; i * i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }
    return is_prime;
}

// Get all primes up to n
vector<int> getPrimes(int n) {
    vector<bool> is_prime = sieve(n);
    vector<int> primes;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) {
            primes.push_back(i);
        }
    }
    return primes;
}
```

### Fast Exponentiation
**Problem:** Calculate a^b mod m efficiently.

**Naive approach:** Multiply a by itself b times - too slow for large b.

**Binary exponentiation:** Use the binary representation of the exponent.

**Example:** a^13 = a^8 × a^4 × a^1 (since 13 = 8 + 4 + 1 in binary)

**How it works:**
1. If exponent is odd, multiply result by current base
2. Square the base and halve the exponent
3. Repeat until exponent becomes 0

#### Fast Exponentiation Implementation
```cpp
long long fastPow(long long base, long long exp, long long mod) {
    long long result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp /= 2;
    }
    return result;
}

// Modular inverse using Fermat's little theorem
long long modInverse(long long a, long long mod) {
    return fastPow(a, mod - 2, mod);  // mod must be prime
}
```

### Matrix Exponentiation
#### Matrix Exponentiation Implementation
```cpp
vector<vector<long long>> multiply(vector<vector<long long>>& A, 
                                  vector<vector<long long>>& B, long long mod) {
    int n = A.size();
    vector<vector<long long>> C(n, vector<long long>(n, 0));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod;
            }
        }
    }
    return C;
}

vector<vector<long long>> matrixPow(vector<vector<long long>>& mat, 
                                   long long exp, long long mod) {
    int n = mat.size();
    vector<vector<long long>> result(n, vector<long long>(n, 0));
    
    // Initialize as identity matrix
    for (int i = 0; i < n; i++) {
        result[i][i] = 1;
    }
    
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = multiply(result, mat, mod);
        }
        mat = multiply(mat, mat, mod);
        exp /= 2;
    }
    return result;
}
```

---

## Part 6: Sorting and Searching

### Binary Search
**Problem:** Find a target value in a sorted array.

**Real-world analogy:** Like guessing a number between 1-100. Always guess the middle, then eliminate half the possibilities.

**Key insight:** Each comparison eliminates half the remaining possibilities.

#### Binary Search Variations
```cpp
// Find first occurrence
int lowerBound(vector<int>& arr, int target) {
    int left = 0, right = arr.size();
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

// Find last occurrence
int upperBound(vector<int>& arr, int target) {
    int left = 0, right = arr.size();
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

// Binary search on answer
bool canAchieve(int value, /* other parameters */) {
    // Check if we can achieve this value
    return true; // or false
}

int binarySearchAnswer(int low, int high) {
    int result = -1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (canAchieve(mid)) {
            result = mid;
            low = mid + 1;  // or high = mid - 1 depending on problem
        } else {
            high = mid - 1;  // or low = mid + 1
        }
    }
    return result;
}
```

### Binary Search on Answer
**Problem type:** "What's the maximum/minimum value such that condition X is satisfied?"

**Examples:**
- "What's the minimum speed to finish a race in time T?"
- "What's the maximum weight we can carry with K trips?"

**Pattern:**
1. Define search space [low, high]
2. Write a function canAchieve(value) that tests if a value works
3. Binary search to find the boundary

### Quick Select (Kth Element)
#### Quick Select Implementation
```cpp
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

int quickSelect(vector<int>& arr, int low, int high, int k) {
    if (low == high) return arr[low];
    
    int pivotIndex = partition(arr, low, high);
    
    if (pivotIndex == k) {
        return arr[pivotIndex];
    } else if (pivotIndex > k) {
        return quickSelect(arr, low, pivotIndex - 1, k);
    } else {
        return quickSelect(arr, pivotIndex + 1, high, k);
    }
}
```

---

## Part 7: Tree Algorithms

### Tree Traversals
**Inorder (Left, Root, Right):** Gives sorted order for BST
**Preorder (Root, Left, Right):** Good for copying/serializing tree
**Postorder (Left, Right, Root):** Good for deleting tree or calculating sizes
**Level order:** Process nodes level by level (BFS on tree)

#### Binary Tree Traversals Implementation
```cpp
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// Inorder traversal
void inorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);
    result.push_back(root->val);
    inorder(root->right, result);
}

// Level order traversal
vector<vector<int>> levelOrder(TreeNode* root) {
    if (!root) return {};
    
    vector<vector<int>> result;
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int size = q.size();
        vector<int> level;
        
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        result.push_back(level);
    }
    return result;
}
```

### LCA (Lowest Common Ancestor)
**Problem:** Find the deepest node that is an ancestor of both given nodes.

**Real-world analogy:** Find the most recent common ancestor in a family tree.

**Approach:** 
1. If current node is one of the target nodes, return it
2. Search left and right subtrees
3. If both subtrees return non-null, current node is the LCA
4. Otherwise, return whichever subtree found a target

#### LCA Implementation
```cpp
TreeNode* lca(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;
    
    TreeNode* left = lca(root->left, p, q);
    TreeNode* right = lca(root->right, p, q);
    
    if (left && right) return root;
    return left ? left : right;
}
```

---

## Part 8: Geometry Basics

### Point and Line Operations
#### Point and Line Implementation
```cpp
struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
    Point operator+(const Point& p) const { return Point(x + p.x, y + p.y); }
    Point operator-(const Point& p) const { return Point(x - p.x, y - p.y); }
    double dot(const Point& p) const { return x * p.x + y * p.y; }
    double cross(const Point& p) const { return x * p.y - y * p.x; }
    double dist() const { return sqrt(x * x + y * y); }
};

// Distance between two points
double distance(Point a, Point b) {
    return (a - b).dist();
}

// Area of triangle using cross product
double triangleArea(Point a, Point b, Point c) {
    return abs((b - a).cross(c - a)) / 2.0;
}
```

---

## Part 9: Why These Algorithms Matter

### Problem Recognition Patterns:
- **Graph connectivity:** Use DFS/BFS or Union-Find
- **Shortest path:** Use Dijkstra or Floyd-Warshall
- **Optimization with constraints:** Often DP
- **Pattern matching:** KMP or similar
- **Finding extremes in sorted data:** Binary search
- **Mathematical calculations:** Fast exponentiation, GCD

### Time Complexity Intuition:
- **O(1):** Array access, hash table operations
- **O(log n):** Binary search, balanced tree operations
- **O(n):** Single pass through data
- **O(n log n):** Efficient sorting, divide-and-conquer
- **O(n²):** Nested loops, some DP problems
- **O(2^n):** Trying all subsets (usually need optimization)

### Contest Strategy Tips:

#### Pattern Recognition is Everything:
- See "shortest path"? Think Dijkstra or BFS
- See "optimize with constraints"? Think DP  
- See "find in sorted data"? Think binary search
- See "connectivity questions"? Think Union-Find or DFS

#### Don't Just Memorize - Understand the "Why":
- Binary search works because it eliminates half the possibilities each time
- DP works because it avoids recalculating the same subproblems
- Dijkstra works because it always processes the closest unvisited node

#### Time Complexity Guidelines for Contests:
- If n ≤ 20: O(2^n) might work (try all subsets)
- If n ≤ 100: O(n³) might work  
- If n ≤ 1000: O(n²) should work
- If n ≤ 100,000: O(n log n) needed
- If n ≤ 1,000,000: O(n) or O(n log n) needed

### Most Critical Algorithms for Contests:

#### Must-Know (80% of problems use these):
1. **DFS/BFS** - Graph traversal and shortest path
2. **Binary Search** - Finding optimal values and searching
3. **Basic DP** - Optimization problems
4. **Sorting algorithms** - Built-in sort() function
5. **Greedy algorithms** - Local optimal choices

#### Important (Next 15% of problems):
1. **Union-Find** - Connectivity and grouping
2. **Dijkstra** - Weighted shortest path
3. **Topological Sort** - Dependency ordering
4. **String algorithms** - Pattern matching
5. **Number theory** - GCD, primes, modular arithmetic

#### Advanced (Remaining 5%):
1. **Floyd-Warshall** - All-pairs shortest path
2. **Advanced DP** - Complex state spaces
3. **Segment trees** - Range queries
4. **Network flow** - Maximum flow problems
5. **Geometry** - Computational geometry

### Contest Problem Types and Algorithm Mapping:

#### Graph Problems:
- **"Find if path exists"** → DFS/BFS
- **"Shortest distance"** → BFS (unweighted) or Dijkstra (weighted)
- **"Connected components"** → DFS or Union-Find
- **"Minimum spanning tree"** → Kruskal's with Union-Find
- **"Topological ordering"** → Topological Sort

#### Optimization Problems:
- **"Maximum/minimum with constraints"** → DP
- **"Subset selection"** → DP or greedy
- **"String matching"** → KMP or built-in find
- **"Range queries"** → Prefix sums or segment trees

#### Search Problems:
- **"Find in sorted array"** → Binary search
- **"Find optimal value"** → Binary search on answer
- **"Kth smallest/largest"** → Quick select or heaps

#### Mathematical Problems:
- **"Large exponents"** → Fast exponentiation
- **"Prime numbers"** → Sieve of Eratosthenes
- **"Divisibility"** → GCD/LCM
- **"Combinatorics"** → DP or mathematical formulas

### Implementation Tips for Contests:

#### Code Templates to Memorize:
```cpp
// Fast I/O
ios_base::sync_with_stdio(false);
cin.tie(NULL);

// Common typedefs
typedef long long ll;
typedef vector<int> vi;
typedef pair<int,int> pii;

// Useful macros
#define FOR(i,a,b) for(int i=(a); i<(b); i++)
#define REP(i,n) FOR(i,0,n)
#define ALL(v) v.begin(), v.end()
```

#### Debugging Techniques:
```cpp
#ifdef LOCAL
    #define dbg(x) cerr << #x << " = " << x << endl
#else
    #define dbg(x)
#endif
```

#### Common Mistakes to Avoid:
1. **Integer overflow** - Use `long long` for large numbers
2. **Array bounds** - Always check indices
3. **Uninitialized variables** - Initialize everything
4. **Wrong data structures** - Choose appropriate containers
5. **Off-by-one errors** - Be careful with loop bounds

### Final Contest Checklist:

#### Before Coding:
1. Read problem completely
2. Identify the algorithm pattern
3. Estimate time complexity
4. Plan your approach
5. Consider edge cases

#### While Coding:
1. Write clean, readable code
2. Use meaningful variable names
3. Add comments for complex logic
4. Test with sample inputs
5. Handle edge cases

#### After Coding:
1. Test with different inputs
2. Check for overflow issues
3. Verify time complexity
4. Look for optimization opportunities
5. Submit when confident

Remember: **Practice makes perfect!** The key to competitive programming success is recognizing patterns quickly and implementing solutions efficiently. Focus on understanding the core concepts rather than just memorizing code.
