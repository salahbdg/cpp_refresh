# C++ Brain Refresh for Competitive Programming & Hackathons

## 1. Basic Syntax & Types

### Fundamental Types
```cpp
int, char, bool, float, double, long, short
auto x = 42;  // type deduction
const int MAX = 100;  // immutable
constexpr int SIZE = 50;  // compile-time constant
```

### Variables & References
```cpp
int x = 10;
int& ref = x;     // reference (alias)
int* ptr = &x;    // pointer
const int* p1;    // pointer to const
int* const p2;    // const pointer
```

## 2. Memory Management

### Stack vs Heap
```cpp
int stack_var = 42;           // automatic storage
int* heap_var = new int(42);  // dynamic allocation
delete heap_var;              // manual cleanup

// Modern approach - smart pointers
std::unique_ptr<int> smart_ptr = std::make_unique<int>(42);
std::shared_ptr<int> shared = std::make_shared<int>(42);
```

### RAII (Resource Acquisition Is Initialization)
- Resources acquired in constructor, released in destructor
- Automatic cleanup when objects go out of scope

## 3. Object-Oriented Programming

### Classes & Objects
```cpp
class MyClass {
private:
    int data;
    
public:
    MyClass(int val) : data(val) {}  // constructor with initializer list
    ~MyClass() {}                    // destructor
    
    int getData() const { return data; }  // const member function
    void setData(int val) { data = val; }
    
    // Copy constructor & assignment operator
    MyClass(const MyClass& other) : data(other.data) {}
    MyClass& operator=(const MyClass& other) {
        if (this != &other) data = other.data;
        return *this;
    }
};
```

### Inheritance & Polymorphism
```cpp
class Base {
public:
    virtual void func() = 0;     // pure virtual (abstract)
    virtual ~Base() = default;   // virtual destructor
};

class Derived : public Base {
public:
    void func() override {}      // override keyword
};
```

## 4. The Rule of Three/Five/Zero

### Rule of Three (C++98)
If you need one, you probably need all three:
- Destructor
- Copy constructor  
- Copy assignment operator

### Rule of Five (C++11)
Add move semantics:
- Move constructor
- Move assignment operator

### Rule of Zero
Prefer using smart pointers and containers to avoid manual resource management

## 5. Templates & Generic Programming

### Function Templates
```cpp
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// Usage
int result = max<int>(5, 10);
auto result2 = max(5.5, 3.2);  // type deduced
```

### Class Templates
```cpp
template<typename T>
class Container {
private:
    T* data;
    size_t size;
    
public:
    Container(size_t s) : size(s), data(new T[s]) {}
    ~Container() { delete[] data; }
    
    T& operator[](size_t index) { return data[index]; }
};
```

## 6. STL (Standard Template Library)

### Containers
```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};
std::map<std::string, int> map;
std::set<int> unique_nums;
std::deque<int> double_ended;
std::list<int> linked_list;
```

### Iterators
```cpp
for (auto it = vec.begin(); it != vec.end(); ++it) {
    std::cout << *it << " ";
}

// Range-based for loop (C++11)
for (const auto& item : vec) {
    std::cout << item << " ";
}
```

### Algorithms
```cpp
#include <algorithm>

std::sort(vec.begin(), vec.end());
auto it = std::find(vec.begin(), vec.end(), 3);
std::transform(vec.begin(), vec.end(), vec.begin(), [](int x) { return x * 2; });
```

## 7. Modern C++ Features (C++11 and beyond)

### Lambda Expressions
```cpp
auto lambda = [](int x, int y) { return x + y; };
auto capture = [&var](int x) { var += x; };  // capture by reference
auto copy_capture = [=](int x) { return var + x; };  // capture by value
```

### Move Semantics
```cpp
class MyClass {
public:
    MyClass(MyClass&& other) noexcept {  // move constructor
        // Transfer resources from other
    }
    
    MyClass& operator=(MyClass&& other) noexcept {  // move assignment
        // Transfer resources from other
        return *this;
    }
};

std::vector<int> vec1 = {1, 2, 3};
std::vector<int> vec2 = std::move(vec1);  // vec1 is now empty
```

### Smart Pointers
```cpp
std::unique_ptr<int> ptr1 = std::make_unique<int>(42);
std::shared_ptr<int> ptr2 = std::make_shared<int>(42);
std::weak_ptr<int> weak = ptr2;  // doesn't affect reference count
```

## 8. Exception Handling

```cpp
try {
    // risky code
    throw std::runtime_error("Something went wrong");
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
} catch (...) {
    std::cerr << "Unknown error" << std::endl;
}
```

## 9. Namespaces

```cpp
namespace MyNamespace {
    void function() {}
}

using namespace std;  // brings entire namespace
using std::cout;      // brings specific symbol
```

## 10. Key Best Practices

### Const Correctness
- Use `const` wherever possible
- Const member functions for accessors
- Pass large objects by const reference

### Initialization
```cpp
int x{42};                    // uniform initialization
std::vector<int> vec{1,2,3};  // initializer list
```

### Resource Management
- Prefer stack allocation
- Use smart pointers for dynamic allocation
- Follow RAII principles

### Performance
- Pass by reference for large objects
- Use move semantics for temporary objects
- Prefer algorithms over raw loops
- Use `const` and `constexpr` for optimization hints

## 11. Common Pitfalls to Avoid

- Forgetting virtual destructors in base classes
- Memory leaks from `new` without `delete`
- Dangling pointers and references
- Copying expensive objects unnecessarily
- Using raw arrays instead of `std::vector`
- Not making single-argument constructors `explicit`

## 12. Competitive Programming Essentials

### Function Parameter Passing (CRUCIAL for contests)
```cpp
// Pass by value - creates copy (slow for large data)
void func1(vector<int> v) {}  // DON'T do this for contests

// Pass by reference - no copy (fast)
void func2(vector<int>& v) {}  // Modifies original

// Pass by const reference - no copy, no modification (fastest for read-only)
void func3(const vector<int>& v) {}  // BEST for reading data

// Arrays decay to pointers
void func4(int arr[], int size) {}  // arr is actually int*
void func5(int arr[100]) {}         // size ignored, still int*
```

### Common Contest Templates & Shortcuts
```cpp
#include <bits/stdc++.h>  // includes everything (contest only!)
using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef vector<vector<int>> vvi;
typedef pair<int,int> pii;

#define FOR(i,a,b) for(int i=(a); i<(b); i++)
#define REP(i,n) FOR(i,0,n)
#define ALL(v) v.begin(), v.end()
#define SZ(v) (int)v.size()

// Fast I/O
ios_base::sync_with_stdio(false);
cin.tie(NULL);
```

### Essential STL for Contests
```cpp
// Vector operations
vector<int> v = {1,2,3};
v.push_back(4);
v.pop_back();
sort(ALL(v));
reverse(ALL(v));
v.erase(v.begin() + 2);  // remove element at index 2

// Set operations (sorted, unique)
set<int> s = {3,1,4,1,5};  // becomes {1,3,4,5}
s.insert(2);
s.erase(3);
bool exists = s.count(4);  // 1 if exists, 0 otherwise

// Map operations
map<string, int> m;
m["key"] = 42;
if (m.count("key")) {}  // check if key exists

// Priority queue (max heap by default)
priority_queue<int> pq;
pq.push(3); pq.push(1); pq.push(4);
int top = pq.top();  // 4
pq.pop();

// Min heap
priority_queue<int, vector<int>, greater<int>> min_pq;
```

### String Manipulation (Contest Frequent)
```cpp
string s = "hello";
s += " world";           // concatenation
s.substr(1, 3);         // "ell" (start=1, length=3)
s.find("ll");           // returns position or string::npos
s.replace(1, 2, "ay");  // "haylo"

// Convert between string and numbers
int num = stoi("123");
string str = to_string(456);

// Character operations
char c = 'A';
bool isLetter = isalpha(c);
bool isDigit = isdigit(c);
char lower = tolower(c);
char upper = toupper(c);
```

### Mathematical Operations
```cpp
#include <cmath>

int result = pow(2, 3);        // 8
int sqrtResult = sqrt(16);     // 4
int absolute = abs(-5);        // 5

// GCD and LCM (C++14+)
int gcd_result = __gcd(12, 18);  // 6
int lcm_result = (a * b) / __gcd(a, b);

// Modular arithmetic (common in contests)
const int MOD = 1e9 + 7;
long long result = (a + b) % MOD;
long long mult = ((long long)a * b) % MOD;
```

### Bit Manipulation (Contest Gold)
```cpp
int x = 5;  // binary: 101

// Check if bit i is set
bool isSet = (x & (1 << i)) != 0;

// Set bit i
x |= (1 << i);

// Clear bit i
x &= ~(1 << i);

// Toggle bit i
x ^= (1 << i);

// Count number of set bits
int count = __builtin_popcount(x);

// Find position of rightmost set bit
int pos = __builtin_ctz(x);  // count trailing zeros
```

### Algorithm Complexity Reminders
```cpp
// Sorting: O(n log n)
sort(v.begin(), v.end());

// Binary search: O(log n)
bool found = binary_search(v.begin(), v.end(), target);
auto it = lower_bound(v.begin(), v.end(), target);

// Set/Map operations: O(log n)
set.insert(), set.find(), map[key]

// Unordered set/map: O(1) average, O(n) worst
unordered_set<int> us;
unordered_map<int, int> um;
```

### Contest-Specific Patterns
```cpp
// Reading unknown number of inputs
int x;
while (cin >> x) {
    // process x
}

// Reading until EOF
string line;
while (getline(cin, line)) {
    // process line
}

// Multiple test cases
int t;
cin >> t;
while (t--) {
    // solve each test case
}

// 2D array initialization
vector<vector<int>> grid(n, vector<int>(m, 0));
```

### Common Contest Algorithms
```cpp
// Binary search template
int binarySearch(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// Two pointers technique
void twoPointers(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left < right) {
        int sum = arr[left] + arr[right];
        if (sum == target) { /* found */ }
        else if (sum < target) left++;
        else right--;
    }
}
```

### Memory & Performance Tips for Contests
```cpp
// Use arrays for better cache performance (if size known)
int arr[100000];  // faster than vector for large sizes

// Reserve vector capacity if you know size
vector<int> v;
v.reserve(100000);

// Use '\n' instead of endl (endl flushes buffer)
cout << result << '\n';  // faster than cout << result << endl;

// Fast input for large datasets
scanf("%d", &n);  // sometimes faster than cin

// Long long for large numbers
long long result = (long long)a * b;  // prevent overflow
```

## 13. Compilation for Contests

```bash
g++ -std=c++17 -O2 -Wall main.cpp -o solution
g++ -DLOCAL -std=c++17 -O2 -Wall main.cpp -o solution  # for local testing
```

### Debug Tips
```cpp
#ifdef LOCAL
    #define dbg(x) cerr << #x << " = " << x << endl
#else
    #define dbg(x)
#endif

dbg(variable);  // prints "variable = value" in local testing only
```
