#!/bin/bash

mogrify -resize 512x512 -gravity center -background black -extent 512x512 -format '512x512.jpg' */*.jpg

## mogrify -shave 100x100 ./*.JPG
