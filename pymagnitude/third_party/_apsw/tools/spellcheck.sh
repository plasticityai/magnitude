#!/bin/sh

# https://github.com/rogerbinns/apsw/pull/229

git grep --color -Eniw '(paramaters|paramater|partifular|seperately|desribes|seperate|seperated|begining|targetted|libary|interpretter|entireity|publically|noone|commiting|statment|simulataneous|exection|wierd|valueable)' | grep -v tools/spellcheck.sh

n=`git grep --color -Eniw '(paramaters|paramater|partifular|seperately|desribes|seperate|seperated|begining|targetted|libary|interpretter|entireity|publically|noone|commiting|statment|simulataneous|exection|wierd|valueable)' | grep -v tools/spellcheck.sh | wc -l`

if [ $n -gt 0 ] ; then exit 1 ; else exit 0 ; fi
