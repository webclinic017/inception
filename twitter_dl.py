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

# get the screen name of all accounts I follow
following = api.GetFriends()
pending_accts = [x.screen_name for x in following]

dl_folder = 'twtr_dl'
os.makedirs(dl_folder, exist_ok=True)

# for all accounts, pickle list of who they follow
friends_list = []
for u in pending_accts:
    print(u)
    they_follow = api.GetFriends(screen_name=u, cursor=-1, )
    print(f'{u} follows {len(they_follow)}')
    friends_list.append(they_follow)
    fname = f'./{dl_folder}/{u}_following.p'
    pickle.dump(they_follow, open(fname, "wb"))
    print(f'Saved: {fname}')

dl_accts = !ls ./{dl_folder}
sn_list = []
for fn in dl_accts:
    u = fn.split('_following')[0]
    following = pickle.load(open(f'{dl_folder}/{fn}', "rb" ))
    sn_list.extend([x.screen_name for x in following])

# load following for each user I follow
# add to "beast" list
index, max_elems = 0, 99
for _ in range(int(len(sn_list) / max_elems) + 1):
    subset = sn_list[index:index + max_elems]
    index += max_elems
    api.CreateListsMember(
        slug='beast', owner_screen_name='vicvveiga', 
        screen_name=subset)