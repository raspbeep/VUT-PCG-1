#!/bin/bash

#
# @file      compare.sh
#
# @author    David Bayer \n
#            Faculty of Information Technology \n
#            Brno University of Technology \n
#            ibayer@fit.vutbr.cz
#
# @brief     PCG Assignment 1
#
# @version   2024
#
# @date      04 October   2023, 09:00 (created) \n
#

if [[ $# < 2 ]]; then
	echo "Usage: <cpu_output.h5> <gpu_output.h5>"
	exit 1
fi

h5diff -v2 -p 0.0001 $1 $2 /pos_x_final
h5diff -v2 -p 0.0001 $1 $2 /pos_y_final
h5diff -v2 -p 0.0001 $1 $2 /pos_z_final
h5diff -v2 -p 0.0001 $1 $2 /vel_x_final
h5diff -v2 -p 0.0001 $1 $2 /vel_y_final
h5diff -v2 -p 0.0001 $1 $2 /vel_z_final
h5diff -v2 -p 0.0001 $1 $2 /weight_final

h5diff -v2 -p 0.0001 $1 $2 /com_x_final
h5diff -v2 -p 0.0001 $1 $2 /com_y_final
h5diff -v2 -p 0.0001 $1 $2 /com_z_final
h5diff -v2 -p 0.0001 $1 $2 /com_w_final

h5diff -v2 -p 0.0001 $1 $2 /pos_x
h5diff -v2 -p 0.0001 $1 $2 /pos_y
h5diff -v2 -p 0.0001 $1 $2 /pos_z
h5diff -v2 -p 0.0001 $1 $2 /vel_x
h5diff -v2 -p 0.0001 $1 $2 /vel_y
h5diff -v2 -p 0.0001 $1 $2 /vel_z
h5diff -v2 -p 0.0001 $1 $2 /weight
h5diff -v2 -p 0.0001 $1 $2 /com_x
h5diff -v2 -p 0.0001 $1 $2 /com_y
h5diff -v2 -p 0.0001 $1 $2 /com_z
h5diff -v2 -p 0.0001 $1 $2 /com_w
