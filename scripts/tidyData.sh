#!/bin/bash

# substitute commas for colons (easier to read into R scripts)
ls * | xargs sed -i "s/,/:/g"

# delete any initial whitespace 
ls * | xargs sed -i "s/^[ \t]*//g"

#delete any empty data
ls * | xargs sed -i '/^$/d'
