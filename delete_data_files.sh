#!/bin/bash

read -r -p "Going to delete datafiles, are you sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    rm -rf Data/network &
    rm -rf Data/ABM &
    rm -rf Data/cfgs
    echo "Deleted datafiles"
else
    echo "Not deleting anything"
fi

read -r -p "Also delete initialized networks? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    rm -rf Data/initialized_network
else
    echo "Not deleting initialized networks"
fi