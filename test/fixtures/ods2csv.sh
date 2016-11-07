#! /usr/bin/env bash
# This script will export all tables from fixtures.ods to separate csv files.
# You will need gnumeric for ssconvert to work!
ssconvert --export-file-per-sheet fixtures.ods %s.csv
