#!/usr/bin/env bash
# Запускаем Jupyter так, чтобы cwd = notebooks
jupyter notebook --notebook-dir="$(pwd)/notebooks" --NotebookApp.default_url='/tree'
