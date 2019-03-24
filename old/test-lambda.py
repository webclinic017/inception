import boto3, json

lambda_client = boto3.client('lambda')

test_event = { "dataset": 'quote/'}

result = lambda_client.invoke(
  FunctionName='list-objects',
  InvocationType='Event',
  Payload=json.dumps(test_event),
)
print(json.load(result['Payload']))
