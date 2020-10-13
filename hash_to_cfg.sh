#!/usr/bin/env bash
hash=$1
db=${2:-db.json}
echo "Querying $db with hash "'"'$hash'"'" yields:"
cat $db | jq ."cfg" | jq .'[] | select(.hash=="'$hash'")'