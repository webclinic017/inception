import twitter
import pandas as pd
import pickle
import os

# auth
CONSUMER_KEY = 'CgmjFmQd7M0LXeWlH7u3SgnoS'
CONSUMER_SECRET = 'Qg94VNKHEysflvd1JOBXTbVc3UiEsIY0pV8Titr55mzn4TERjh'
ACCESS_TOKEN = '310273490-rOhArEbc0kuSjBbU2zHErgaCG4Xp9m0E4fDxY5AG'
ACCESS_TOKEN_SECRET = 'Ew8IJEhsozmzwW4oxprzdKR4puqJv05vFg8QRrPNtW0X6'

api = twitter.Api(consumer_key=CONSUMER_KEY,
                  consumer_secret=CONSUMER_SECRET,
                  access_token_key=ACCESS_TOKEN,
                  access_token_secret=ACCESS_TOKEN_SECRET,
                  sleep_on_rate_limit=True)

iam_following = api.GetFriends()

pending_accts = ['cdixon',
                 'joulee',
                 'DavidSacks',
                 'rabois',
                 'nntaleb',
                 'benedictevans',
                 'mcuban',
                 'bgarlinghouse',
                 'aantonop',
                 'JamesClear',
                 'PTetlock',
                 'EricTopol',
                 'patrickc',
                 'AnnieDuke',
                 'podcastnotes',
                 'EricRWeinstein',
                 'm2jr',
                 'wolfejosh',
                 'soumithchintala',
                 'wintonARK',
                 'lloydblankfein',
                 'masason',
                 'sbarnettARK',
                 'zarazhangrui',
                 'Arie_Belldegrun',
                 'FEhrsam',
                 'LauraDeming',
                 'PalmerLuckey',
                 'fredwilson',
                 'moskov',
                 'paulg',
                 'sapinker',
                 'stratechery',
                 'vkhosla',
                 'ylecun',
                 'eladgil',
                 'bgurley',
                 'bhavanaYarasuri',
                 'jwangARK',
                 'msamyARK',
                 'skorusARK',
                 'TashaARK',
                 'finkd',
                 'kaifulee',
                 'Chad_Hurley',
                 'eldsjal',
                 'leijun',
                 'ShouZiChew',
                 'JohnathanIve',
                 'RobertIger',
                 'tim_cook',
                 'BillGates',
                 'hanstung',
                 'davidein',
                 'WarrenBuffett',
                 'benbernanke',
                 'LHSummers',
                 'HowardMarksBook',
                 'BillAckman',
                 'JTLonsdale',
                 'woodhaus2',
                 'RobertGreene',
                 'sparker',
                 'tfadell',
                 'thielfellowship',
                 'moritzKBE',
                 'gdb',
                 'briansin',
                 'naval',
                 'pmarca',
                 'bhorowitz',
                 'KenHowery',
                 'mlevchin',
                 'karpathy',
                 'demishassabis',
                 'travisk',
                 'TruthGundlach',
                 'chamath',
                 'RayDalio',
                 'sama',
                 'reedhastings',
                 'AndrewYNg',
                 'JeffBezos',
                 'peterthiel',
                 'elonmusk',
                 'foundersfund']

dl_folder = 'twtr_dl'
os.makedirs(dl_folder, exist_ok=True)

friends_list = []
for u in pending_accts:
    they_follow = api.GetFriends(screen_name=u, cursor=-1, )
    print(f'{u} follows {len(they_follow)}')
    friends_list.append(they_follow)
    fname = f'./{dl_folder}/{u}_following.p'
    pickle.dump(they_follow, open(fname, "wb"))
    print(f'Saved: {fname}')
