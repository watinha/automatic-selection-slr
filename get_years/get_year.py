import re, sys, json

import bibtexparser as bibparser
import urllib.parse as parse

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

bibs = [
    '../bibs/ontologies/round1-google.bib',
    '../bibs/ontologies/round2-ieee.bib',
    '../bibs/ontologies/round3-google.bib',
    '../bibs/ontologies/round1-ieee.bib',
    '../bibs/ontologies/round1-outros.bib',
    '../bibs/ontologies/round2-google.bib',
    '../bibs/mdwe/round1-sciencedirect.bib',
    '../bibs/mdwe/round1-acm.bib',
    '../bibs/mdwe/round1-ieee.bib',
    '../bibs/xbi/round1-google.bib',
    '../bibs/xbi/round2-ieee.bib',
    '../bibs/xbi/round3-google.bib',
    '../bibs/xbi/round1-ieee.bib',
    '../bibs/xbi/round1-outros.bib',
    '../bibs/xbi/round2-google.bib',
    '../bibs/illiterate/round1-others.bib',
    '../bibs/testing/round1-google.bib',
    '../bibs/testing/round2-ieee.bib',
    '../bibs/testing/round3-google.bib',
    '../bibs/testing/round2-outros.bib',
    '../bibs/testing/round1-ieee.bib',
    '../bibs/testing/round1-outros.bib',
    '../bibs/testing/round2-google.bib',
    '../bibs/slr/round1-todos.bib',
    '../bibs/games/round1-todos.bib',
    '../bibs/pair/round1-todos.bib'
]

json_file = open('result.json')
jsons = json_file.read()
json_file.close()
title_years = json.loads(jsons)

def main():
    re_year = re.compile('\d{4}')
    for bib in bibs:
        bibfile = bib
        print('=== Updating bibs available in %s ===' %
                (bibfile))
        f = open(bibfile)
        db = bibparser.load(f)

        count = 0
        years = []
        for entry in db.entries:
            try:
                year = entry['year']
                years.append(int(year))
                pass
            except:
                count += 1
                year = re_year.search(title_years[entry['title']])
                if year == None:
                    print('%d: %s -> %s' % (count, entry['title'], 'none'))
                else:
                    print('%d: %s -> %s' % (count, entry['title'], year.group(0)))
                entry['year'] = str(year.group(0))

        if count > 0:
            new_f = open(bibfile, 'w')
            bibparser.dump(db, new_f)

        print(sorted(years))


if __name__ == '__main__':
    main()
