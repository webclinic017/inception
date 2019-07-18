#!/bin/bash

echo "Quotes for the last three days"
aws s3 ls fund-fin-data/quote/ | tail -n 3 #last 3 dates for quotes

echo "Count summary objects for $(date +"%Y-%m-%d")"
eval aws s3 ls fund-fin-data/summary/$(date +"%Y-%m-%d")/ | wc #count objects for a date > 1257 companies

echo "List summary category overrides"
aws s3 ls fund-fin-data/summary-categories/ #list last date of overrides

echo "Count daily pricing objects for $(date +"%Y-%m-%d")"
eval aws s3 ls fund-fin-data/pricing/1d/ | grep $(date +"%Y-%m-%d") | wc #count pricing objects for a date > 1304 companies

# echo "Count minute pricing objects for $(date +"%Y-%m-%d")"
# eval aws s3 ls fund-fin-data/pricing/1m/ | grep $(date +"%Y-%m-%d") | wc #count pricing objects for a date > 1304 companies

echo "Count valuation waterfall objects"
aws s3 ls fund-fin-data/valuation/waterfall/ | tail -n 3 #last 3 dates for quotes

echo "List ML recommend objects"
echo "macro_ML"
aws s3 ls fund-fin-data/recommend/macro_ML/ | tail -n 3
echo "micro_ML"
aws s3 ls fund-fin-data/recommend/micro_ML/ | tail -n 3
echo "marketcap_ML"
aws s3 ls fund-fin-data/recommend/marketcap_ML/ | tail -n 3
echo "equity portfolio"
aws s3 ls fund-fin-data/recommend/equities/ | tail -n 3
