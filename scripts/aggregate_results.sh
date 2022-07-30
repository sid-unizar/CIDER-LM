#!/bin/zsh

TRACKDIR="1-3-val_1"
ALIGNMENT_NUMER="363" # 362 + 1

incorrect=0
# Move
for f in ciderlm/results/*; do
	if [ `ls $f/$TRACKDIR/ | wc -l` -ne $ALIGNMENT_NUMER ]; then
		echo "Wrong number of final alignments!!!!!!!! in $f"
		incorrect=1
	else
		cp "$f/$TRACKDIR/aggregated/Main/aggregatedPerformance.csv" "results-aggregated/results_`echo $f | cut -d ' ' -f 1 --complement `.csv"
	fi
done

if [ $incorrect -eq 1 ]; then
	#exit 1
fi

# Get header
for f in results-aggregated/results_*; do
	echo `cat $f | head -1`
	break
done > results-aggregated/aggregated-total.csv

# Get results with version as header 
for f in results-aggregated/results_*; do
	echo `cat $f | head -2 | tail -1` | sed --expression="s/ALL/`echo $f | cut -d '_' -f 1 --complement | rev | cut -d '.' -f 1 --complement | rev`/g"
done >> results-aggregated/aggregated-total.csv
