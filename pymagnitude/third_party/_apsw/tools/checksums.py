#!/usr/bin/env python
#
# See the accompanying LICENSE file.
#
import urllib2
import hashlib
import re

sqlitevers=(
    '3250200',
    '3250100',
    '3250000',
    '3240000',
    '3230100',
    '3230000',
    '3220000',
    '3210000',
    '3200100',
    '3200000',
    '3190300',
    '3190200',
    '3190100',
    '3190000',
    '3180000',
    '3170000',
    '3160200',
    '3160100',
    '3160000',
    '3150200',
    '3150100',
    '3150000',
    '3140100',
    '3140000',
    '3130000',
    )

# Checks the checksums file

def getline(url):
    for line in open("checksums", "rtU"):
        line=line.strip()
        if len(line)==0 or line[0]=="#":
            continue
        l=[l.strip() for l in line.split()]
        if len(l)!=4:
            print "Invalid line in checksums file:", line
            raise ValueError("Bad checksums file")
        if l[0]==url:
            return l[1:]
    return None

def check(url, data):
    d=["%s" % (len(data),), hashlib.sha1(data).hexdigest(), hashlib.md5(data).hexdigest()]
    line=getline(url)
    if line:
        if line!=d:
            print "Checksums mismatch for", url
            print "checksums file is", line
            print "Download is", d
    else:
        print url,
        if url.endswith(".zip"):
            print "  ",
        print d[0], d[1], d[2]

# They keep messing with where files are in URI - this code is also in setup.py
def fixup_download_url(url):
    ver=re.search("3[0-9]{6}", url)
    if ver:
        ver=int(ver.group(0))
        if ver>=3071600:
            if ver>=3220000:
                year="2018"
            elif ver>=3160000:
                year="2017"
            elif ver>=3100000:
                year="2016"
            elif ver>=3080800:
                year="2015"
            elif ver>=3080300:
                year="2014"
            else:
                year="2013"
            if "/"+year+"/" not in url:
                url=url.split("/")
                url.insert(3, year)
                return "/".join(url)
    return url

for v in sqlitevers:
    # Windows amalgamation
    AURL="https://sqlite.org/sqlite-amalgamation-%s.zip" % (v,)
    AURL=fixup_download_url(AURL)
    try:
        data=urllib2.urlopen(AURL).read()
    except:
        print AURL
        raise
    check(AURL, data)
    # All other platforms amalgamation
    AURL="https://sqlite.org/sqlite-autoconf-%s.tar.gz" % (v,)
    AURL=fixup_download_url(AURL)
    try:
        data=urllib2.urlopen(AURL).read()
    except:
        print AURL
        raise
    check(AURL, data)
