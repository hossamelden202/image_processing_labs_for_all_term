# Lab 1: Python Fundamentals and Basic Syntax

## Overview

Lab 1 serves as the foundational introduction to Python programming for image processing. This laboratory covers essential Python syntax, data structures, and programming concepts necessary for implementing image processing algorithms. Students will learn core Python fundamentals that form the foundation for subsequent image processing laboratories.

This laboratory emphasizes practical understanding through hands-on coding experience rather than abstract theory. The focus is on developing fluency with Python syntax and understanding how fundamental programming concepts apply to numerical and image data.

## Detailed Description

### Python Language Fundamentals

#### Variable Declaration and Dynamic Typing

Python uses dynamic typing, where variables are declared without explicit type specifications:

```python
a = 3              # Integer
b = True           # Boolean
c = 'String!'      # String
```

**Key Concepts:**
- Variables are created on first assignment
- Type is inferred from the assigned value
- The same variable can be reassigned to different types
- No type declarations needed
- Type can be checked with `type()` function

**Data Types:**
- **int**: Whole numbers (no size limit in Python 3)
- **float**: Decimal numbers with finite precision
- **bool**: Boolean values (True or False)
- **str**: Text strings enclosed in quotes
- **complex**: Complex numbers with real and imaginary parts
- **None**: Special null value representing absence of value

#### Arithmetic Operations

Python provides standard mathematical operations:

```python
a = 5
b = 2

print(5 / 2)    # Float division: 2.5
print(5 // 2)   # Integer division: 2
print(5 % 2)    # Modulo (remainder): 1
print(2 ** 3)   # Exponentiation: 8

# divmod returns tuple (quotient, remainder)
print(divmod(5, 2))  # (2, 1)
```

**Important Distinctions:**
- `/` always produces float result in Python 3
- `//` performs integer division (floor division)
- `%` computes remainder after division
- `**` performs exponentiation (not `^`)
- `divmod()` simultaneously computes quotient and remainder

#### Boolean Operations and Conditionals

Conditional logic controls program flow based on boolean conditions:

```python
# Simple conditional
if condition:
    # Execute if condition is True
    
# If-else structure
if condition:
    # Execute if True
else:
    # Execute if False

# Multiple conditions
if condition1:
    # Execute if condition1 True
elif condition2:
    # Execute if condition1 False and condition2 True
else:
    # Execute if all previous conditions False
```

**Comparison Operators:**
- `==`: Equal to
- `!=`: Not equal to
- `<`: Less than
- `>`: Greater than
- `<=`: Less than or equal to
- `>=`: Greater than or equal to

**Logical Operators:**
- `and`: Both conditions must be True
- `or`: At least one condition must be True
- `not`: Negates boolean value

### Array and List Operations

#### Python Lists

Lists are ordered, mutable collections:

```python
arr = [1, 2, 3, 4, 5]
print(arr[0])      # First element: 1
print(arr[-1])     # Last element: 5
print(arr[1:4])    # Slice (indices 1, 2, 3): [2, 3, 4]
print(arr[::2])    # Every second element: [1, 3, 5]
```

**List Properties:**
- 0-indexed (first element at index 0)
- Negative indices count from end (-1 is last element)
- Slicing creates new list with subset of elements
- Range notation: start:stop:step

**List Modification:**
```python
arr = [1, 2, 3, 4, 5]
arr[0] = 10         # Modify single element
arr[1:3] = [20, 30] # Modify slice
arr.append(6)       # Add element to end
arr.extend([7, 8])  # Add multiple elements
```

#### NumPy Arrays

NumPy provides efficient numerical arrays essential for image processing:

```python
import numpy as np

# Create arrays
zeros = np.zeros((3, 4))      # 3x4 array of zeros
ones = np.ones((3, 4))        # 3x4 array of ones
arr = np.array([1, 2, 3, 4])  # Create from list

# Array properties
print(arr.shape)    # Dimensions: (4,)
print(arr.dtype)    # Data type: int64
print(arr.size)     # Total elements: 4
```

**Array Advantages over Lists:**
- Uniform data type (all elements same type)
- Efficient memory usage
- Vectorized operations (operate on entire array)
- Mathematical operations implemented in compiled code
- Essential for image processing performance

#### Conditional Indexing

NumPy arrays support powerful conditional selection:

```python
arr = np.array([1, 2, 3, 4, 5])

# Boolean mask
mask = arr > 2
print(arr[mask])    # [3 4 5]

# Multiple conditions
mask = (arr > 1) & (arr < 5)
print(arr[mask])    # [2 3 4]

# Negation
mask = ~(arr == 3)
print(arr[mask])    # [1 2 4 5]
```

**Use Cases in Image Processing:**
- Select pixels meeting intensity criteria
- Create binary masks from thresholds
- Modify regions based on conditions
- Extract specific pixel ranges

### Control Flow: Loops

#### For Loops

Iterate over sequences:

```python
# Iterate over list
arr = [1, 2, 3, 4, 5]
for x in arr:
    print(x)

# Iterate with indices
for i in range(len(arr)):
    print(arr[i])

# Enumerate to get both index and value
for i, x in enumerate(arr):
    print(f"Index {i}: Value {x}")
```

**Range Function:**
- `range(n)`: 0 to n-1 (n total values)
- `range(start, stop)`: start to stop-1
- `range(start, stop, step)`: With custom step size

#### Nested Loops

Essential for 2D operations like image processing:

```python
# 2D array iteration
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

for row in matrix:
    for value in row:
        print(value)

# Using range for indices (important for images)
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        print(matrix[i][j])
```

**Image Processing Connection:**
- Images are 2D arrays (grayscale) or 3D arrays (color)
- Processing requires nested loops over dimensions
- Vectorized NumPy operations often replace explicit loops

#### While Loops

Execute while condition is true:

```python
counter = 0
while counter < 5:
    print(counter)
    counter += 1
```

### Functions and Methods

#### Defining Functions

Functions encapsulate reusable code blocks:

```python
def function_name(parameter1, parameter2):
    """Function docstring describing purpose and parameters"""
    # Function body
    result = parameter1 + parameter2
    return result

# Call function
output = function_name(3, 4)
```

**Function Components:**
- **Name**: Identifies function for calling
- **Parameters**: Input values (optional)
- **Docstring**: Documentation of purpose
- **Body**: Code that executes when called
- **Return**: Value(s) to return (optional)

#### Keyword Arguments

Python supports flexible parameter passing:

```python
def function(x, y, z):
    print(x, y, z)

# Positional arguments
function(1, 2, 3)

# Keyword arguments
function(z=3, y=2, x=1)  # Order doesn't matter

# Mixed
function(1, z=3, y=2)
```

**Advantages:**
- Makes code more readable
- Parameter order doesn't matter with keywords
- Allows default parameter values
- Improves function flexibility

#### Default Parameters

Functions can have default values:

```python
def greet(name="World"):
    print(f"Hello, {name}!")

greet()              # Uses default: "Hello, World!"
greet("Alice")       # Overrides default: "Hello, Alice!"
```

### Special Python Concepts

#### None Value

Python uses `None` to represent absence of value (equivalent to null in other languages):

```python
x = None
if x is None:
    print("x is None")

# Useful for initialization
result = None
if some_condition:
    result = compute_value()
```

#### String Formatting

Modern f-strings provide clean string formatting:

```python
name = "Alice"
age = 30

# F-string (Python 3.6+)
print(f"{name} is {age} years old")

# Format method
print("{} is {} years old".format(name, age))

# Old-style (not recommended)
print("%s is %d years old" % (name, age))
```

#### List Comprehensions

Concise syntax for creating lists:

```python
# Traditional approach
squares = []
for i in range(10):
    squares.append(i**2)

# List comprehension
squares = [i**2 for i in range(10)]

# With condition
even_squares = [i**2 for i in range(10) if i % 2 == 0]
```

## Key Learning Objectives

Upon completion of this laboratory, students will:

1. Understand Python's dynamic typing system
2. Perform arithmetic operations correctly
3. Use conditional logic to control program flow
4. Work with lists and NumPy arrays effectively
5. Implement loops for iterative processing
6. Write and call functions
7. Use conditional indexing for array manipulation
8. Understand None and special values
9. Apply Python to numerical computations
10. Transition smoothly to image processing tasks

## How to Run the Laboratory

### Prerequisites

Install required Python packages:

```bash
pip install numpy matplotlib scikit-image
```

### Execution Steps

1. Navigate to the Lab 1 directory:
```bash
cd Lab1
```

2. Open the Jupyter notebook:
```bash
jupyter notebook "Python Basic Syntax.ipynb"
```

3. Execute cells in sequence to explore:
   - Variable declaration and types
   - Arithmetic operations (division, modulo, divmod)
   - Loop structures (for, while)
   - Conditional logic (if, elif, else)
   - Array indexing and slicing
   - NumPy array creation and operations
   - Function definition and calling
   - Conditional selection using boolean masks

4. Experimentation:
   - Modify variable values and observe behavior
   - Try different array operations
   - Create your own functions
   - Explore conditional logic branches
   - Practice loop constructs

5. Extension activities:
   - Write functions for mathematical computations
   - Implement nested loops for 2D operations
   - Create conditional logic for decision-making
   - Explore edge cases and error handling

### Expected Outputs

- Console output from executed code cells
- Array and list manipulations visualized
- Function results and computations
- Conditional logic demonstrations
- Loop iteration results

## Common Pitfalls and Important Notes

### Division Operator

In Python 3, `/` always returns float:
- `5 / 2 = 2.5` (not 2)
- Use `//` for integer division: `5 // 2 = 2`

### Array Indexing

Remember zero-based indexing:
- First element is at index 0
- Last element is at index -1
- Slicing excludes the stop index: `arr[0:3]` includes indices 0, 1, 2

### Boolean Operations

Use `and`/`or` instead of C-style `&&`/`||`:
```python
# Correct Python
if x > 0 and y < 10:

# Incorrect
if x > 0 && y < 10:  # SyntaxError
```

### None Comparisons

Use `is None` instead of `== None`:
```python
# Preferred
if x is None:

# Works but not recommended
if x == None:
```

### Loop Variable Scope

Loop variables persist after loop completion:
```python
for i in range(5):
    pass
print(i)  # i still equals 4
```

## Laboratory Files

- `Python Basic Syntax.ipynb`: Main notebook with Python fundamentals
- `lol1.ipynb`: Additional practice exercises
- `histogram/`: Directory for histogram-related examples
- `hsv/`: Directory for HSV color space examples

## References and Resources

- Python official documentation: https://docs.python.org
- NumPy documentation: https://numpy.org
- Real Python tutorials: https://realpython.com
- Python for Data Science: https://www.continuum.io

## Notes for Students

- Python syntax emphasizes readability and simplicity
- Consistent indentation is mandatory (not optional)
- NumPy is essential for efficient image processing
- List comprehensions make code concise and fast
- Comments and docstrings improve code understanding
- Test code incrementally as you write
- Use debugging tools to understand program flow
- Practice basic concepts before advanced topics
