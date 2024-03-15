# coding=utf-8
import aiohttp
import asyncio
import json
import os
import sqlite3
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from config import SQLITE_DB_DIR, SQLITE_DB_NAME, MAX_CRAWLER_REQUESTS, DOWNLOAD_HTML_DIR, MAX_EMBEDDING_INPUT


class AsyncCrawler:

    def __init__(self, base_url, sqlite_db_path=f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}", max_requests=MAX_CRAWLER_REQUESTS, download_html_dir=DOWNLOAD_HTML_DIR):
        self.base_url = self.normalize_url(base_url)
        self.sqlite_db_path = sqlite_db_path
        self.visited_urls = set()
        self.semaphore = asyncio.Semaphore(max_requests)
        self.count = 0
        self.download_html_dir = download_html_dir
        if not os.path.exists(download_html_dir):
            os.makedirs(download_html_dir)
    
    async def fetch_page(self, session, url):
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    return await response.text()
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                return None

    def is_same_domain(self, url):
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def normalize_url(self, url):
        return url.split('#')[0].rstrip('/')

    async def save_content_to_db_and_disk(self, url, content):
        trans_url = url.replace('/', '_')
        # Download html to local disk
        file_path = os.path.join(self.download_html_dir, f"{trans_url}.html")
        with open(file_path, 'w') as fd:
            fd.write(content)
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO t_raw_tab (url, content, content_type, content_length, crawl_time) VALUES (?, ?, ?, ?, ?)',
                  (url, content, 'text/html', len(content), int(time.time())))
        conn.commit()
        conn.close()

    async def parse_links(self, session, html, url):
        #text, links = await self.parse_content(html, url)
        text, links = await self.parse_content_v2(html, url)
        await self.save_content_to_db_and_disk(url, text)
        for full_link in links:
            normalized_link = self.normalize_url(full_link)
            if normalized_link not in self.visited_urls and self.is_same_domain(normalized_link):
                self.visited_urls.add(normalized_link)
                await self.crawl(session, normalized_link)

    async def parse_content(self, html, base_url):
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title text
            title_text = soup.title.string.strip() if soup.title else ""
            
            # Extract body text
            body_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
            
            # Extract multimedia descriptions
            multimedia_descs = []

            # Extract descriptions from video elements
            for video in soup.find_all('video'):
                if video.get('title'):
                    multimedia_descs.append(video['title'].strip())
                for track in video.find_all('track'):
                    if track.get('kind') == 'descriptions' and track.get('label'):
                        multimedia_descs.append(track['label'].strip())

            # Extract titles from object, embed, and iframe elements
            for obj in soup.find_all(['object', 'embed', 'iframe']):
                if obj.get('title'):
                    multimedia_descs.append(obj['title'].strip())
            
            multimedia_descs_str = " ".join(multimedia_descs)
            
            # Combine title text, body text, and multimedia descriptions
            full_text = " ".join(filter(None, [title_text, body_text, multimedia_descs_str]))
            
            # Extract links within the body
            links = {urljoin(base_url, a['href']) for a in soup.find_all('a', href=True)} if soup.body else set()
            
            return full_text, links
        except Exception as e:
            print(f"Error processing content from {base_url}: {str(e)}")
            return "", set()

    async def parse_content_v2(self, html, base_url):
        chunk_text_vec = []
        max_token_len = MAX_EMBEDDING_INPUT
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Extract links within the body
            links = {urljoin(base_url, a['href']) for a in soup.find_all('a', href=True)} if soup.body else set()

            # Extract title text
            title_text = soup.title.string.strip() if soup.title else ""
            curr_chunk = title_text + '\n'
            curr_len = len(title_text)

            # Remove <script> and <style> elements before extracting text
            for script_or_style in soup(['script', 'style', 'head', 'meta', 'link']):
                script_or_style.decompose()

            last_element_text = ''
            if soup.body:
                for element in soup.body.descendants:
                    element_text = ''
                    if element.name == 'a' and element.get('href'):
                        href = element['href']
                        # Check if href is an external link
                        if href.startswith('http://') or href.startswith('https://'):
                            text = element.get_text(strip=True)
                            # Embed the link directly in the text
                            element_text = f"{text}[{href}]"
                        else:
                            # For relative URLs, just include the text
                            element_text = element.get_text(strip=True)
                    elif element.string and not element.name:
                        # Append non-link text
                        element_text = element.string.strip()

                    if element_text:
                        element_len = len(element_text)
                        if element_len > max_token_len:
                            print(f"warning element_len={element_len}")

                        if curr_len + element_len <= max_token_len:
                            curr_chunk += element_text
                            curr_chunk += '\n'
                            curr_len += element_len + 1
                        else:
                            chunk_text_vec.append(curr_chunk)
                            curr_chunk = last_element_text + '\n' + element_text + '\n'
                            curr_len = len(curr_chunk)
                        last_element_text = element_text

                if curr_chunk:
                    chunk_text_vec.append(curr_chunk)

            full_text = json.dumps(chunk_text_vec)

            return full_text, links
        except Exception as e:
            print(f"Error processing content from {base_url}: {str(e)}")
            return "", set()

    async def crawl(self, session, url):
        self.count += 1
        print(f"Count: {self.count}")
        print(f"Crawling: {url}")
        html = await self.fetch_page(session, url)
        if html:
            await self.parse_links(session, html, url)

    async def run(self):
        async with aiohttp.ClientSession() as session:
            self.visited_urls.add(self.base_url)
            await self.crawl(session, self.base_url)


if __name__ == "__main__":
    #base_url = "https://www.openim.io/en"
    #base_url = "https://docs.openim.io/"

    url_vec = ["https://www.openim.io/en", "https://docs.openim.io/"]
    for base_url in url_vec:
        print(f"base_url={base_url}")
        begin = int(time.time())
        crawler = AsyncCrawler(base_url)
        asyncio.run(crawler.run())
        end = int(time.time())

        time_cost = end - begin
        print(f"\ntime_cost={time_cost}")

