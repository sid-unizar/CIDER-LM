#!/bin/zsh

# Move
for f in ciderlm/results/results_*; do
	cp "$f/multifarm_all-v2/aggregated/Main/aggregatedPerformance.csv" "results-aggregated/results_`echo $f | cut -d ' ' -f 1 --complement `.csv"
done

# Get header
for f in results-aggregated/results_*; do
	echo `cat $f | head -1`
	break
done > results-aggregated/aggregated-total.csv

# Get results with version as header 
for f in results-aggregated/results_*; do
	echo `cat $f | head -2 | tail -1` | sed --expression="s/ALL/`echo $f | cut -d '_' -f 1 --complement | rev | cut -d '.' -f 1 --complement | rev`/g"
done >> results-aggregated/aggregated-total.csv
