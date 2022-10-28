#!/bin/sh

prefix() {
    while read line; do
      echo "$1$line"
    done
}

join() {
    acc=""
    while read line; do
      acc="$acc$1$line"
    done
    echo "$acc"
}

python_dirs="$(\
  find . -name '*.py' -type f \
    | xargs -I {} dirname {} \
    | uniq \
    | grep -v '^.$' \
    | cut -c 3- \
    | prefix "$(pwd)/" \
    | join ':')"
export PYTHONPATH="$([ -n "$PYTHONPATH" ] && echo "$PYTHONPATH:" || echo '')$(pwd):$python_dirs"
exec python3 $*