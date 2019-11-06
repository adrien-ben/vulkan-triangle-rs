#!/bin/bash

find ./assets/shaders -not -name *.spv -type f -exec glslangValidator -V -o {}.spv {} \;
