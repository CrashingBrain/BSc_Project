#!/bin/bash

# comment out all lipsum
perl -p -i -e 's/(?<!%)(\\lipsum)/%\2/g' `find ./ -name "*.tex"`
