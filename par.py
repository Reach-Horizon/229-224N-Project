# -*- coding: utf-8 -*-
"""
    par.py: Check a text file for matching sets of parentheses, brackets,
    and braces
    
    Copyright (C) 2013  Greg von Winckel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Last updated: Mon Oct 14 14:07:32 MDT 2013
"""

import sys

# Correspondence between opening and closing symbols
pairing =  {'(':')','[':']','{':'}'}

# Reverse lookup version of pairing for describing mismatches
reverse  = dict((value,key) for key,value in pairing.items())

# Set of valid symbols of interest
valid = [item for items in pairing.items() for item in items]
 

class counter(object):
    def __init__(self):
        # Most recently thing opened has to be the next thing closed
        self.lastopened = []
        self.ok = True

    def update(self,char):
        # If we have an opening character
        if char in pairing.keys():
            self.lastopened.append(char)

        # If a closing character is encountered
        else:
            # Make sure that there has at least been an 
            # opening character 
            if len(self.lastopened)<1:
                print('Character ' + char + ' encountered before ' + \
                      reverse[char])
                self.ok = False
            else:
             # Make sure the closing character corresponds to the 
             # opening character 
                if char != pairing[self.lastopened[-1]]:
                    self.ok = False
                    print('Closing character ' + \
                           pairing[self.lastopened[-1]] + \
                           ' expected, but ' + char + \
                           ' was encountered first')

                # If the pair is properly closed, delete the 
                # last opening symbol 
                else: self.lastopened.pop()

            


if __name__ == '__main__':

    C = counter()

    # Read in the text file line by line
    for line in open(sys.argv[1],'r').readlines():

        # Scan line for valid characters
        for k in range(len(line)):
            if line[k] in valid:
                C.update(line[k])
                                
    if len(C.lastopened) > 0:
        print("Reached EOF without closing character " + \
               pairing[C.lastopened[-1]])
    else:
        if C.ok is True:
            print("Parenthesis, Braces, and " +\
                  "Brackets are matched correctly")
