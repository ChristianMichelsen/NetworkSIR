#!/bin/bash

read -r -p "Going to delete datafiles, are you sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    rm -rf Data/network &
    pids[0]=$!
    rm -rf Data/ABM &
    pids[1]=$!
    rm -rf Data/cfgs
    echo "Deleted datafiles"
else
    echo "Not deleting anything"
fi

read -r -p "Also delete initialized networks? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    rm -rf Data/initialized_network
    echo "Deleted initialized networks"
else
    echo "Not deleting initialized networks"
fi

read -r -p "Also delete database file (db.json)? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    rm db.json
    echo "Deleted database file"
else
    echo "Not deleting database file"
fi


# wait for all pids
for pid in ${pids[*]}; do
    echo "waiting for PID = ${pid} to finish"
    wait $pid
done
