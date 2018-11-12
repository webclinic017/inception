import basic_utils.py
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):

    symbol_list = event['symbols']
    dataset = 'quote'

    result = [x.key for x in bucket.objects.filter(Prefix=dataset)]

    full_data = []
    index, max_elems = 0, MAX_SYMBOLS

    for q in range(int(len(symbol_list) / max_elems) + 1):
        subset = symbol_list[index:index + max_elems]
        index += max_elems
        symbols = get_cs_tickers(subset)
        logger.info('Getting quotes for {}'.format(symbols))
        encoded_kv = urllib.parse.urlencode(
            {QUERY_DICT[dataset][enc_key]: symbols}
        )
        data = get_data(encoded_kv, dataset, '')
        full_data.extend(get_children_list(json.loads(data), 'quoteResponse'))
    data = json.dumps(full_data)
    path = get_path(dataset)

    objectName = path + json_ext.format(str(today_date))
    logger.info('Saving results at {}'.format(objectName))
    store_s3(data, objectName)
