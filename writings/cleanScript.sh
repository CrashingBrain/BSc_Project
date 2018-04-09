#!/bin/bash

# comment lipsum
perl -p -i -e 's/(?<!%)(\\lipsum)/%\2/g' `find ./ -name "*.tex"`

# uncomment lipsum
# perl -p -i -e 's/%(\\lipsum)/\1/g' `find ./ -name "*.tex"`
