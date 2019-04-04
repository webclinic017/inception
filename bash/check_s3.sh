#!/bin/bash

echo "Quotes for the last three days"
aws s3 ls fund-fin-data/quote/ | tail -n 3 #last 3 dates for quotes

echo "Count summary objects for $(date +"%Y-%m-%d")"
eval aws s3 ls fund-fin-data/summary/$(date +"%Y-%m-%d")/ | wc #count objects for a date > 1257 companies

echo "List summary category overrides" 
aws s3 ls fund-fin-data/summary-categories/ #list last date of overrides

echo "Count daily pricing objects for $(date +"%Y-%m-%d")"
eval aws s3 ls fund-fin-data/pricing/1d/ | grep $(date +"%Y-%m-%d") | wc #count pricing objects for a date > 1304 companies

echo "Count minute pricing objects for $(date +"%Y-%m-%d")"
eval aws s3 ls fund-fin-data/pricing/1m/ | grep $(date +"%Y-%m-%d") | wc #count pricing objects for a date > 1304 companies

echo "Count valuation water objects"
aws s3 ls fund-fin-data/valuation/waterfall/ | tail -n 3 #last 3 dates for quotes

echo "List ML recommend objects"
aws s3 ls fund-fin-data/recommend/ #last 3 dates for quotes
