#! /bin/bash

### QUESTION 1 ####

# Given images
python q1.py -i "data/q1/chess1.jpg" -a "data/annotation/q1_annotation.npy" -o "output"
python q1.py -i "data/q1/facade.jpg" -a "data/annotation/q1_annotation.npy" -o "output"
python q1.py -i "data/q1/tiles5.jpg" -a "data/annotation/q1_annotation.npy" -o "output"

# My Collected Images
# python q1.py -i data/q1/chess1.png -a data/annotation/q1_annotation.npy -o output
# python q1.py -i data/q1/chess1.png -a data/annotation/q1_annotation.npy -o output