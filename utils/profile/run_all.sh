#!/bin/bash

for d in ./mrcnn/*; do
  if [ -d "$d" ]; then         # or:  if test -d "$d"; then
    ( cd "$d" && ./driver.sh )
  fi
done

for d in ./watershed/*; do
  if [ -d "$d" ]; then         # or:  if test -d "$d"; then
    ( cd "$d" && ./driver.sh )
  fi
done
