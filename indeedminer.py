from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import urlencode
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys, re, string, datetime

if (len(sys.argv) < 3): 
    print("\n\tUsage: indeedminer.py <search keywords> <location> <optional: search page count>")
    print('\te.g. $pythonw indeedminer.py "HR Manager" "New York"\n')
    exit()

search_page = 1
if (len(sys.argv) > 3):
    search_page = int(sys.argv[3]) 

search_keyword= sys.argv[1]
location = sys.argv[2]
params = {
        'q':search_keyword,
        'l':location
    }

#replace url_prefix with your favorite country from https://www.indeed.com/worldwide
url_prefix = "https://www.indeed.com"    
url = url_prefix + "/jobs?"+urlencode(params)

def getJobLinksFromIndexPage(soup): 
    jobcards = soup.find_all('div', {'class':'jobsearch-SerpJobCard row result'})
    
    job_links_arr = []
    
    #get job links
    for jobcard in tqdm(jobcards): 
        job_title_obj = jobcard.find('a', {'class':'turnstileLink'})
        job_title_link = job_title_obj.get('href')
        job_links_arr.append(job_title_link)
        
    return job_links_arr

def getJobInfoLinks(url, next_page_count, url_prefix):
    job_links_arr = []
   
    while True: 
        if (next_page_count < 1):
            break      
        
        next_page_count -= 1
        
        html = urlopen(url)
        soup = BeautifulSoup(html, 'lxml')
        
        job_links_arr += getJobLinksFromIndexPage(soup)

        pagination = soup.find('div', {'class':'pagination'})  
        next_link = ""
        for page_link in reversed(pagination.find_all('a')):
            #reserve the pagination array to load the last element
            next_link_idx = page_link.get_text().find("Next")
            if (next_link_idx >= 0):
                 next_link = page_link.get('href')   
                 break 
         
        if (next_link == ""):
            break
        
        url = url_prefix+next_link           
            
    return job_links_arr   

current_datetime = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
print("Getting job links in {} page(s)...".format(search_page))
job_links_arr = getJobInfoLinks(url, search_page, url_prefix)

#if no nltk, go download
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

punctuation = string.punctuation
job_desc_arr=[] 
print("Getting job details in {} post(s)...".format(len(job_links_arr)))
for job_link in tqdm(job_links_arr): 
    job_link = url_prefix+job_link
    #print("Handling->{}".format(job_link))
    job_html = urlopen(job_link)
    job_soup = BeautifulSoup(job_html, 'lxml')
    job_desc = job_soup.find('div', {'class':'jobsearch-JobComponent-description'})
    job_meta = job_desc.find('div', {'class':'jobsearch-JobMetadataHeader-item'})
    #remove job meta
    if (job_meta is not None): 
        job_meta.decompose()
    #add a space before each <li> to add spacing
    for li_tag in job_desc.findAll('li'):
        li_tag.insert(0, " ")   
    job_desc = job_desc.get_text()    
    #remove http    
    job_desc = re.sub('https?:\/\/.*[\r\n]*', '', job_desc, flags=re.MULTILINE)
    #replace punctutaion to space
    job_desc = job_desc.translate(job_desc.maketrans(punctuation, ' ' * len(punctuation)))    
    job_desc_arr.append(job_desc)

stop_words =  stopwords.words('english')   
extra_stop_words = ["experience", "position", "work", "please", "click", "must", "may", "required", "preferred", 
                    "type", "including", "strong", "ability", "needs", "apply", "skills", "requirements", "company", 
                    "knowledge", "job", "responsibilities", location.lower()] + location.lower().split()
stop_words += extra_stop_words

print("Generating Word Cloud...")
#TFIDF
tfidf_para = {
    "stop_words": stop_words,
    "analyzer": 'word',   #analyzer in 'word' or 'character' 
    "token_pattern": r'\w{1,}',    #match any word with 1 and unlimited length 
    "sublinear_tf": False,  #False for smaller data size  #Apply sublinear tf scaling, to reduce the range of tf with 1 + log(tf)
    "dtype": int,   #return data type 
    "norm": 'l2',     #apply l2 normalization
    "smooth_idf":False,   #no need to one to document frequencies to avoid zero divisions
    "ngram_range" : (1, 2),   #the min and max size of tokenized terms
    "max_features": 500    #the top 500 weighted features
}
tfidf_vect = TfidfVectorizer(**tfidf_para)
transformed_job_desc = tfidf_vect.fit_transform(job_desc_arr)

#Generate word cloud
freqs_dict = dict([(word, transformed_job_desc.getcol(idx).sum()) for word, idx in tfidf_vect.vocabulary_.items()])
w = WordCloud(width=800,height=600,mode='RGBA',background_color='white',max_words=500).fit_words(freqs_dict)
plt.figure(figsize=(12,9))
plt.title("Keywords:[{}] Location:[{}] {}".format(search_keyword,location, current_datetime))
plt.imshow(w)
plt.axis("off")
plt.show()