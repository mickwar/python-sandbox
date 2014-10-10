#################################

# import math
# OR
# from math import *

import random
from math import *


# Strings 
myString = "Hello World"
print myString

# Variables
x = 2
y = 3

print x + y

# Arrays

arr = [1, 2, 3, 4]
mat = [[1, 2], [3, 4]]

print arr

# Loops

for x in arr:
    print x

for x in mat:
    print x

# Conditionals
# Note that python cares about spacing. no braces

if myString == "Hello World":
    print "yes you are right"
else:
    print "no you are wrong"

# Functions

def callMom():
    print "hi mom how are you?"

callMom()

# classes

class Human():
    def __init__(self, name):
        self.name  = name
        self.speed = 0
    def run(self):
        self.speed = self.speed + 1
    def whatsmyspeed(self):
        print self.speed
    def whatsmyname(self):
        print self.name 


bryan = Human("bryan")

bryan.whatsmyspeed()
bryan.run()

for n in range(1,10):
    bryan.run()
    bryan.whatsmyspeed()

bryan.whatsmyname()
