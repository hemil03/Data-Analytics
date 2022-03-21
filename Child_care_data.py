from bs4 import BeautifulSoup
import pandas as pd
from requests import get
from time import sleep, time
import re

names = []     #list for storing names of child care
address = []    #list for storing address of child care
contact = []    #list for storing contact number of child care
c_url = []      #list for storing website of child care
cor_url = []    #list for storing corrected website of child care
alphabets = map(chr, range(97, 123))    #list for storing ahphabates for searching alphabatically

for i in alphabets:
    #searching childcre alphabatically 
    url = ('https://www.toronto.ca/data/children/dmc/a2z/a2z'+str(i)+'.html')
    resp = get(url)
    bs = BeautifulSoup(resp.text, 'html.parser')
    
    #finding and storing name of childcare
    c_links = bs.find('tbody').find_all('a')
    for c_name in c_links:
        names.append(c_name.text)
    
    #finding and storing website of childcare
    care_links = bs.find("tbody").find_all("a",attrs={'href': re.compile("^")})
    for care_link in care_links:
        c_url.append(care_link.get('href'))

#removing errors in collected urls 
for i in range(len(c_url)):
    cor_url.append(c_url[i].replace('..','https://www.toronto.ca/data/children/dmc'))

#collecting contact info and address of childcare from corrected urls 
for i in cor_url:
    url = (str(i))
    resp = get(url)
    bs = BeautifulSoup(resp.text, 'html.parser')
    c_add = bs.find('div', class_='csd_opcrit_content_box').find('header').find('p')
    address.append(c_add.text)
    c_cont = bs.find('ul').find('li', class_='nudge')
    contact.append(c_cont.text)

#removing errors in collected addresses
add = []
for i in range(len(address)):
    add.append(address[i].replace('\n','').replace('\t\xa0',''))

#removing errors in collected contact info
c_cont = []
for i in range(len(contact)):
    contact[i] = contact[i].replace('Phone:\n\t\t\t','').replace('\xa0\t\t\t\t','').replace('\t','').replace('\n','')
    c_cont.append(re.sub(' +', ' ', contact[i]))

#creating Pandas dataframe for storing information
df_care = list(zip(names, add, c_cont))
child_care = pd.DataFrame(df_care, columns = ['Care_Name', 'Address', 'Contact'])

#converting collected data into .csv file
child_care.to_csv('child_care.csv')