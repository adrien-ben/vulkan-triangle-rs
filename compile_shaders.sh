#!/bin/bash

find ./shaders -not -name *.spv -type f -exec glslangValidator -V -o {}.spv {} \;
