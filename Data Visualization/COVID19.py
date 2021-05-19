# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:08:35 2020

@author: L01506162
"""
#Librerías importadas  
import requests
from bs4 import BeautifulSoup
#import pickle

try: 
	from googlesearch import search 
except ImportError: 
	print("No module named 'google' found") 

# to search 
query = "Twitter covid19"

for j in search(query, tld="co.in", num=10, stop=10, pause=2): 
	print(j) 

#Transcript URL
def url_to_transcript(j):
    '''Returns transcript data specifically from scrapsfromtheloft.com.'''
    page = requests.get(j).text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.find(class_="post-content").find_all('p')]
    print(j)
    return text

transcripts = [url_to_transcript(u) for u in j]

