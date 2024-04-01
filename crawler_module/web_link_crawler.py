# coding=utf-8
import aiohttp
import aiosqlite
import asyncio
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from utils.logger_config import my_logger as logger


class AsyncCrawlerSiteLink:

    def __init__(self, base_url, sqlite_db_path, max_requests, version, redis_lock):
        logger.info(f"[CRAWL_LINK] init, base_url:'{base_url}', version:{version}")
        self.base_url = self.normalize_url(base_url)
        self.sqlite_db_path = sqlite_db_path
        self.visited_urls = set()
        self.semaphore = asyncio.Semaphore(max_requests)
        self.domain = urlparse(self.base_url).netloc
        self.version = version
        self.redis_lock = redis_lock
        self.count = 0

    async def fetch_page(self, session, url):
        logger.info(f"[CRAWL_LINK] fetch_page, url:'{url}'")
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    return await response.text()
            except Exception as e:
                logger.error(f"[CRAWL_LINK] fetch_page, Error fetching {url}: {e}")
                return None

    def is_same_domain(self, url):
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def normalize_url(self, url):
        return url.split('#')[0].rstrip('/')

    async def save_link_to_db(self, url):
        logger.info(f"[CRAWL_LINK] save_link_to_db, url:'{url}'")
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            # Check if the URL already exists
            cursor = await db.execute('SELECT id FROM t_raw_tab WHERE url = ?', (url,))
            result = await cursor.fetchone()

            timestamp = int(time.time())
            if await self.redis_lock.aacquire_lock():
                try:
                    if result:
                        # Extract the id from the result
                        record_id = result[0]
                        # If exists, update doc_status to 1, and also update version and mtime using id
                        logger.info(f"[CRAWL_LINK] save_link_to_db, url:'{url}' is existed in t_raw_tab, doc_id:{record_id}, update version:{self.version}")
                        await db.execute(
                            'UPDATE t_raw_tab SET doc_status = ?, version = ?, mtime = ? WHERE id = ?',
                            (1, self.version, timestamp, record_id)
                        )
                    else:
                        # If not exists, insert a new record
                        logger.info(f"[CRAWL_LINK] save_link_to_db, url:'{url}' is not existed in t_raw_tab, version:{self.version}")
                        await db.execute(
                            'INSERT INTO t_raw_tab (domain, url, content, content_length, doc_status, version, ctime, mtime) '
                            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                            (self.domain, url, '', 0, 1, self.version, timestamp, timestamp)
                        )
                    await db.commit()
                finally:
                    await self.redis_lock.arelease_lock()

    async def parse_link(self, session, html, url):
        logger.info(f"[CRAWL_LINK] parse_link, url:'{url}'")
        link_vec = await self.extract_link(html, url)
        await self.save_link_to_db(url)
        for full_link in link_vec:
            normalized_link = self.normalize_url(full_link)
            if normalized_link not in self.visited_urls and self.is_same_domain(normalized_link):
                self.visited_urls.add(normalized_link)
                await self.crawl_link(session, normalized_link)

    async def extract_link(self, html, url):
        logger.info(f"[CRAWL_LINK] extrack_link, url:'{url}'")
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # Extract links within the body
            link_vec = {urljoin(url, a['href']) for a in soup.find_all('a', href=True)} if soup.body else set()
            return link_vec
        except Exception as e:
            logger.error(f"Error processing content from {url}: {str(e)}")
            return set()

    async def crawl_link(self, session, url):
        self.count += 1
        logger.info(f"[CRAWL_LINK] craw_link, url:'{url}', count:{self.count}")
        html = await self.fetch_page(session, url)
        if html:
            await self.parse_link(session, html, url)

    async def update_site_domain_status(self, domain_status):
        logger.info(f"[CRAWL_LINK] update_site_domain_status, domain_status:{domain_status}")
        timestamp = int(time.time())
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            if await self.redis_lock.aacquire_lock():
                try:
                    await db.execute(
                        "UPDATE t_domain_tab SET domain_status = ?, mtime = ? WHERE domain = ?",
                        (domain_status, timestamp, self.domain)
                    )
                    await db.commit()
                finally:
                    await self.redis_lock.arelease_lock()

    async def mark_expired_link(self):
        logger.info(f"[CRAWL_LINK] mark_expired_link")
        timestamp = int(time.time())
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            if await self.redis_lock.aacquire_lock():
                try:
                    # Update doc_status to 5 for URLs that are not currently marked as status 1
                    await db.execute(
                        "UPDATE t_raw_tab SET doc_status = ?, mtime = ? WHERE domain = ? AND doc_status != 1",
                        (5, timestamp, self.domain)
                    )
                    await db.commit()
                finally:
                    await self.redis_lock.arelease_lock()

    async def run(self):
        logger.info(f"[CRAWL_LINK] run begin!")
        async with aiohttp.ClientSession() as session:
            self.visited_urls.add(self.base_url)
            await self.crawl_link(session, self.base_url)

        await self.mark_expired_link()
        await self.update_site_domain_status(2)
        logger.info(f"[CRAWL_LINK] run end!")

