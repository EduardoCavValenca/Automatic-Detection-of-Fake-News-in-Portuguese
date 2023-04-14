import requests
import pandas as pd
import json
from datetime import datetime
from bs4 import BeautifulSoup
import threading
import re
import demoji
from time import sleep, perf_counter
from datetime import timedelta

bad_paragraphs = ['Leia também', 'Leia aqui', 'Leia mais', 'Leia outr', 'Veja também', 'Veja aqui', 'Veja mais', 'Veja outr', 'Compartilhe no ', 'Compartilhe na ', 'Compartilhe nos ', 'Compartilhe nas ', 'Compartilhe esta ', 'Compartilhe estas ', 'Compartilhe este ', 'Compartilhe isto ', 'Siga-nos', 'Inscreva-se', 'Últimas notícias', 'Acesse também', 'Descubra mais', 'Não perca', 'Acesse agora', 'Confira', 'Foto: ', 'Fotos: ', 'Reprodução: ', 'Vídeos: ', 'Vídeo: ']
kill_all_threads = False

def process_text(text):
    # remove all \n, \t and \r
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')

    # remove single quotes and double quotes
    text = text.replace("'", '').replace('"', '')

    # Remove emojis
    text = demoji.replace(text, '')

    # remove all multiple spaces
    text = re.sub(r' +', ' ', text)

    # replace the - with the unicode character for – (the former breaks DptOIE)
    text = text.replace('-', u'\u2013') 
    
    return text.lower()

def request_news(page: int) -> json:
    try:
        response = requests.get(f'https://falkor-cda.bastian.globo.com/tenants/g1/instances/4af56893-1f9a-4504-9531-74458e481f91/posts/page/{page}'.format(page))
        return response.json()
    except Exception as e:
        print("error requesting news list: ", e)
        return None

def get_full_news(url: str) -> str:
    # Get the html of the url
    news_html = None
    try:
        news_html = requests.get(url).text
    except Exception as e:
        print("Error getting news html: ", e)
        return ''
    soup = BeautifulSoup(news_html, 'html.parser')

    # Get all the paragraphs in divs with id starting with chunk- and 5 alphanumeric characters
    paragraphs = soup.find_all('div', {'id': re.compile(r'chunk-\w{5}')})

    # remove paragraphs that have words with all caps (usually links to other news)
    paragraphs = [p for p in paragraphs if not any([bad_paragraph.lower() in p.get_text().lower() for bad_paragraph in bad_paragraphs])]

    # Get the text of all the paragraphs
    full_text = ' '.join([p.getText(separator=' ').strip() for p in paragraphs])

    return process_text(full_text)

def filter_news_request_response(response: json, our_news) -> list:
    # Filter the response to get only the news that have a title, url, section and content
    news = []
    if not response:
        return news

    for item in response['items']:
        title = None
        section = None
        date = None
        url = None
        content = None
        try:
            url = item['content']['url']
            # check if the url is already in this thread's news
            for single_news in our_news:
                if single_news['url'] == url:
                    #print(f'News already in thread: {url}. Skipping...')
                    continue

            title = item['content']['title']
            if any([bad_title.lower() in title.lower() for bad_title in bad_paragraphs]):
                continue
            title = process_text(title)

            section = process_text(item['content']['section'])

            try:
                date = datetime.strptime(item['publication'], r'%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d')
            except:
                date = datetime.strptime(item['publication'], r'%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')

            content = get_full_news(item['content']['url'])
        except:
            continue
        
        if title and section and date and url and content and len(title) > 30 and len(section) > 2 and len(date) > 2 and len(url) > 5 and len(content) > 30:
            news.append({
                'title': title,
                'section': section,
                'date': date,
                'url': url,
                'content': content,
            })
        else:
            #print(f'Error getting news: [TITLE: {title}, SECTION: {section}, DATE: {date}, URL: {url}, CONTENT: {content}]')
            pass
        
    return news

def build_csv(max_number_of_news_per_thread, thread_index: int, news: list):
    
    # Build the csv file
    # each thread will access a different part of the news list
    # thread 0 will access the first max_number_of_news news
    # thread 1 will access the next max_number_of_news news
    # and so on
    # the last thread will access the last max_number_of_news news
    start_index = thread_index * max_number_of_news_per_thread
    end_index = start_index + max_number_of_news_per_thread

    news_per_page = 10
    number_of_pages_per_thread = max_number_of_news_per_thread // news_per_page
    start_page = thread_index * number_of_pages_per_thread + 1
    end_page = start_page + number_of_pages_per_thread + 1
    print(f'[Thread {thread_index:02d}] Scanning news pages [{start_page:04d}-{end_page:04d}) for indexes [{start_index:05d}-{end_index:05d})...')
    page = start_page

    this_news = [] * max_number_of_news_per_thread
    this_news_index = 0
    i = 0
    global kill_all_threads
    while kill_all_threads is False:
        # Log the progress
        i += 1
        if i % 10 == 0:
            print(f'[Thread {thread_index:02d}] Currently on Page {page:5d} - {len(this_news):4d} news scraped so far')

        # Get a page of news
        response = request_news(page)
        new_news = filter_news_request_response(response, this_news)
        page += 1
        
        # Save the page of news
        prev_news_index = this_news_index
        this_news_index += min(len(new_news), max_number_of_news_per_thread - this_news_index)
        this_news[prev_news_index:this_news_index] = new_news[0:this_news_index - prev_news_index]
        news[start_index + prev_news_index:start_index + this_news_index] = new_news[0:this_news_index - prev_news_index]

        if this_news_index == max_number_of_news_per_thread:
            print(f'[Thread {thread_index:02d}] Got the maximum number of news ({max_number_of_news_per_thread:04d}). Exiting...')
            break

        if page == end_page:
            print(f'[Thread {thread_index:02d}] Could not get enough news. Got {len(this_news):04d} news')
            break

def number_of_unique_news(news, number_of_news, time_perf_start):
    news_not_none = [n for n in news if n]
    unique = len(set([n['url'] for n in news_not_none]))
    total = len(news_not_none)
    repeated = total - unique
    print(f'[MASTER] Got {unique:05d}/{number_of_news:05d} unique news so far ({unique/number_of_news:.2%}) --- {repeated:05} repeated ({repeated/(total if total > 0 else 1):.2%}). {timedelta(seconds=(perf_counter() - time_perf_start))}')
    return unique

def monitor_threads(news, number_of_news, time_perf_start):
    global kill_all_threads
    while kill_all_threads is False:
        # Check number of unique news
        unique_news_count = number_of_unique_news(news, number_of_news, time_perf_start)

        # Check if we have enough news
        if unique_news_count >= number_of_news:
            print(f'[MASTER] Got {unique_news_count} news, enough to stop. Waiting for threads to finish...')
            kill_all_threads = True

        sleep(30)

def input_to_kill_early():
    global kill_all_threads
    while kill_all_threads is False:
        input('[MASTER] Press enter to kill all threads...\n')
        are_you_sure = input('[MASTER] Are you sure you want to kill all threads? (y/n) ')
        if are_you_sure.lower() == 'y':
            print('[MASTER] Killing all threads...')
            kill_all_threads = True

if __name__ == '__main__':
    number_of_threads = 24
    number_of_news = 10000

    number_of_news_per_thread = number_of_news // number_of_threads + 1

    # Assume that we are going to get at least 3 unique news per page (kinda optimistic)
    min_news_per_page = 3
    max_number_of_news_per_thread = number_of_news_per_thread * min_news_per_page
    max_number_of_news = max_number_of_news_per_thread * number_of_threads

    threads = []
    news = [None] * max_number_of_news * number_of_threads
    print(f'[MASTER] Will get {number_of_news} news using {number_of_threads} threads')
    time_perf_start = perf_counter()
    for i in range(number_of_threads):
        t = threading.Thread(target=build_csv, args=(max_number_of_news_per_thread, i, news))
        t.start()
        threads.append(t)

    t = threading.Thread(target=monitor_threads, args=(news, number_of_news, time_perf_start))
    t.start()
    threads.append(t)
    
    t = threading.Thread(target=input_to_kill_early)
    t.start()
    threads.append(t)

    for t in threads:
        t.join()

    print('[MASTER] Hooray! All threads finished')

    # remove None values
    news = [n for n in news if n]

    # print number of unique news
    number_of_unique_news(news, number_of_news, time_perf_start)


    # Create a dataframe with the news
    df = pd.DataFrame(news)
    df.to_csv('./data/csvs/g1_news.csv', index=False)

    print('[MASTER] Done!')