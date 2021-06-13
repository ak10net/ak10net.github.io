---
layout: post
title: "Inner functions"
author: "Ankit"
tags: python
excerpt_separator: <!--more-->
---

### Let's understand python inner functions.<!--more-->

If you have seen Inception movie understanding inner functions would be much easy. Similar to dream with a dream concept from inception, we have function within a function in Python. Like in the movie inception to get in and out of dream external stimulus was needed. Similarly python nested functions have direct access to variables and names (stimulus) in enclosing function. Like in the movie dreams exhibited certain characteristics such as 'no access to outer world while in a dream state', 'dream can be shared with other people', 'change in inner dream could alter reality in outer world' and many more. Python inner functions share some of those characteristics. Let's take brief tour of type of inner functions and why we need them.

```python
# Simple inner function
def outer(who):
  def inner():
    print(f"Hello, {who}")
  inner()

outer('world!')
Hello, World!
```

### Why do we need inner functions though ?

+ To provide encapsulation and hide functions from external access

Provide encapsulation like a parent. This means that you can't access children function. When trying to access you will get name error saying there is no such function. Because parent function has hidden its child function

```python
def outer(who):
  def inner():
    print(f"Hello, {who}")
  inner() 

inner()
NameError: name 'inner' is not defined
```

+ Facilitate code reuse by writing helper functions

Like counting most occurring value in a data frame column or string independent of the type of input.

```python
from collections import Counter
def process_outer(x):
  def process_inner(y):
    counter = Counter(y)
    print(f'Most common word is {counter.most_common(1)[0][0]}')   

  if isinstance(x, str):
    z = list(x.split(','))
    process_inner(z)
  else:
    process_inner(x)   

process_outer(['rat','cat','bat','sat','rat','rat','cat','rat','sat'])
Most common word is rat
process_outer('rat,cat,bat,sat,cat')
Most common word is cat
```

+ Create Closures (better remember them as state aware functions)

All functions are of equal order but some are higher-order functions. Higher order functions can take other functions as arguments, return them or both. The difference between closure and nested functions above is that retain state even after the enclosing function has finished executing. In essence closure function are remember variables of enclosing function even after enclosing functions has executed.

```python
# Simple closure function
def total_sales(units):
  def sales(price):
    return units * price
  return sales

price_two = total_sales(2)
price_two(4)
8
```

+ Create decorators (not interior decorators although their work is similar)

Decorators like closure are higher order functions which can be called as an argument and they return another callable. Main aim of decorator functions is to add new functionality to an existing function without changing their definition. Much like you call interior decorator at home to improve home and add new vibe to it. Like in code below a normal greet function is decorated by add_greetings.

```python
# decorator function
def add_greetings(func):
  def _add_greetings():
    print("Hello")
    func()
    print("Good bye!")
  return _add_greetings

@add_greetings
def greet():
  print('I am decorated with greetings!')

greet()
Hello
I am decorated with greetings!
Good Bye!
```

*Pretty interesting and fun right!*