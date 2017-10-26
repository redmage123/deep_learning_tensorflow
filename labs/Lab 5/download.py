#!/usr/bin/env python3

import urllib.request


pos_file = urllib.request.urlopen('https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.pos').read()

with open('data/rt-polarity.pos','w') as out:
    out.write(str(pos_file))

neg_file = urllib.request.urlopen('https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.neg').read()
with open('data/rt-polarity.neg','w') as out:
    out.write(str(neg_file))



