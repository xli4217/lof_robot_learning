#!/bin/bash

COMPUTE_LOCATION=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[ 0]}" )" && pwd )"


export COPPELIASIM_ROOT=$DIR/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04/
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

export PYTHONPATH=$DIR/libs/spinningup:$DIR/libs/RLBench/:$DIR/libs/:$PYTHONPATH
export PKG_PATH=$DIR
