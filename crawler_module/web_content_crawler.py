# coding=utf-8
import aiohttp
import aiosqlite
import asyncio
import json
import os
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from utils.logger_config import my_logger as logger


class AsyncCrawlerSiteContent:

    def __init__(self, domain, sqlite_db_path, max_requests, max_embedding_input, document_embedder_obj, redis_lock):
        logger.info(f"[CRAWL_CONTENT] init, domain:'{domain}'")
        self.domain = domain
        self.sqlite_db_path = sqlite_db_path
        self.semaphore = asyncio.Semaphore(max_requests)
        self.max_embedding_input = max_embedding_input
        self.document_embedder_obj = document_embedder_obj
        self.redis_lock = redis_lock
        self.count = 0

    async def fetch_page(self, session, doc_id, url):
        logger.info(f"[CRAWL_CONTENT] fetch_page, doc_id:{doc_id}, url:'{url}'")
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    return await response.text()
            except Exception as e:
                logger.error(f"[CRAWL_CONTENT] fetch_page, Error fetching doc_id:{doc_id}, url:'{url}', exception:{e}")
                return None

    async def parse_content(self, html, url):
        logger.info(f"[CRAWL_CONTENT] parse_content, url:'{url}'")
        chunk_text_vec = []
        max_token_len = self.max_embedding_input
        try:
            soup = BeautifulSoup(html, 'html.parser')

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
                            logger.warn(f"[CRAWL_CONTENG] parse_conteng, url:'{url}', warning element_len={element_len}")

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

            return chunk_text_vec
        except Exception as e:
            logger.error(f"[CRAWL_CONTENG] parse_content, url:'{url}', Error processing content exception:{str(e)}")
            return []

    async def save_content_to_db(self, html, doc_id, url):
        logger.info(f"[CRAWL_CONTENT] save_content_to_db, doc_id:{doc_id}, url:'{url}'")
        chunk_text_vec = await self.parse_content(html, url)
        content = json.dumps(chunk_text_vec)
        timestamp = int(time.time())

        existing_content = None
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            # Fetch the existing content for comparison
            cursor = await db.execute(
                'SELECT content FROM t_raw_tab WHERE id = ?', (doc_id,))
            existing_row = await cursor.fetchone()
            existing_content = existing_row[0] if existing_row else None

        # Compare the new content with the existing content
        if content != '' and existing_content != content:
            logger.info(f"[CRAWL_CONTENT] save_content_to_db, doc_id:{doc_id}, url:'{url}', content is updated")
            # If content has changed, update the content and content_length in t_raw_tab
            async with aiosqlite.connect(self.sqlite_db_path) as db:
                # Enable WAL mode for better concurrency
                await db.execute("PRAGMA journal_mode=WAL;")

                if await self.redis_lock.aacquire_lock():
                    try:
                        await db.execute(
                            'UPDATE t_raw_tab SET content = ?, content_length = ?, doc_status = ?, mtime = ? WHERE id = ?',
                            (content, len(content), 3, timestamp, doc_id)
                        )
                        await db.commit()
                    finally:
                        await self.redis_lock.arelease_lock()

            logger.info(f"[CRAWL_CONTENT] save_content_to_db, doc_id:{doc_id}, url:'{url}', delete_embedding_doc")
            await self.delete_embedding_doc([doc_id])

            data = [(doc_id, url, chunk_text_vec)]
            logger.info(f"[CRAWL_CONTENT] save_content_to_db, doc_id:{doc_id}, url:'{url}', document_embedder_obj.aadd_content_embedding")
            records_to_add, records_to_update = await self.document_embedder_obj.aadd_content_embedding(data)
            if records_to_add:
                # Insert new embedding records and update t_raw_tab doc_status to 4
                async with aiosqlite.connect(self.sqlite_db_path) as db:
                    # Enable WAL mode for better concurrency
                    await db.execute("PRAGMA journal_mode=WAL;")

                    if await self.redis_lock.aacquire_lock():
                        try:
                            await db.executemany(
                                "INSERT INTO t_doc_embedding_map_tab (doc_id, embedding_id_list, ctime, mtime) VALUES (?, ?, ?, ?)",
                                records_to_add
                            )
                            await db.executemany("UPDATE t_raw_tab SET doc_status = ?, mtime = ? WHERE id = ?", records_to_update)
                            await db.commit()
                        finally:
                            await self.redis_lock.arelease_lock()
        else:
            logger.info(f"[CRAWL_CONTENT] save_content_to_db, doc_id:{doc_id}, url:'{url}', content is not updated")
            async with aiosqlite.connect(self.sqlite_db_path) as db:
                # Enable WAL mode for better concurrency
                await db.execute("PRAGMA journal_mode=WAL;")

                if await self.redis_lock.aacquire_lock():
                    try:
                        # If content hasn't changed, directly update doc_status to 4 without changing content
                        await db.execute(
                            'UPDATE t_raw_tab SET doc_status = ?, mtime = ? WHERE id = ?',
                            (4, timestamp, doc_id)
                        )
                        await db.commit()
                    finally:
                        await self.redis_lock.arelease_lock()

    async def crawl_content(self, session, doc_id, url):
        self.count += 1
        logger.info(f"[CRAWL_CONTENT] crawl_content, doc_id:{doc_id}, url:'{url}', count:{self.count}")

        # Update doc_status to 2 before fetching the page
        await self.update_doc_status(doc_id, 2)

        html = await self.fetch_page(session, doc_id, url)
        if html:
            await self.save_content_to_db(html, doc_id, url)

    async def update_doc_status(self, doc_id, doc_status):
        logger.info(f"[CRAWL_CONTENT] update_doc_status, doc_id:{doc_id}, doc_status:{doc_status}")
        timestamp = int(time.time())
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            if await self.redis_lock.aacquire_lock():
                try:
                    await db.execute(
                        "UPDATE t_raw_tab SET doc_status = ?, mtime = ? WHERE id = ?",
                        (doc_status, timestamp, doc_id)
                    )
                    await db.commit()
                finally:
                    await self.redis_lock.arelease_lock()

    async def add_content(self, url_dict):
        logger.info(f"[CRAWL_CONTENT] add_content begin!, url_dict:{url_dict}")
        async with aiohttp.ClientSession() as session:
            task_vec = [self.crawl_content(session, doc_id, url_dict[doc_id]) for doc_id in url_dict]
            await asyncio.gather(*task_vec)

        await self.check_and_update_domain_status()
        logger.info(f"[CRAWL_CONTENT] add_content end!, url_dict:{url_dict}")

    async def delete_embedding_doc(self, doc_id_vec):
        logger.info(f"[CRAWL_CONTENT] delete_embedding_doc, doc_id_vec:{doc_id_vec}")
        doc_id_tuple = tuple(doc_id_vec)
        placeholder = ','.join('?' * len(doc_id_vec))  # Create placeholders
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            cursor = await db.execute(f"SELECT embedding_id_list FROM t_doc_embedding_map_tab WHERE doc_id IN ({placeholder})", doc_id_tuple)
            rows = await cursor.fetchall()
            # Parse embedding_id_list and flatten the list
            embedding_id_vec = [id for row in rows for id in json.loads(row[0])]
            if embedding_id_vec:
                logger.info(f"[CRAWL_CONTENT] delete_embedding_doc, document_embedder_obj.delete_content_embedding:{embedding_id_vec}")
                self.document_embedder_obj.delete_content_embedding(embedding_id_vec)

            if await self.redis_lock.aacquire_lock():
                try:
                    # Delete records from t_doc_embedding_map_tab
                    await db.execute(f"DELETE FROM t_doc_embedding_map_tab WHERE doc_id IN ({placeholder})", doc_id_tuple)
                    await db.commit()
                finally:
                    await self.redis_lock.arelease_lock()

    async def delete_content(self, url_dict, delete_raw_table=True):
        logger.info(f"[CRAWL_CONTENT] delete_content begin, url_dict:{url_dict}, delete_row_table:{delete_raw_table}")
        # Convert doc_id list to a tuple for SQL execution
        doc_id_vec = list(url_dict.keys())
        doc_id_tuple = tuple(doc_id_vec)
        placeholder = ','.join('?' * len(doc_id_vec))  # Create placeholders
        await self.delete_embedding_doc(doc_id_vec)

        if delete_raw_table:
            # After deleting embeddings, delete records from t_raw_tab
            async with aiosqlite.connect(self.sqlite_db_path) as db:
                # Enable WAL mode for better concurrency
                await db.execute("PRAGMA journal_mode=WAL;")

                if await self.redis_lock.aacquire_lock():
                    try:
                        # Delete records from t_raw_tab
                        await db.execute(f"DELETE FROM t_raw_tab WHERE id IN ({placeholder})", doc_id_tuple)
                        await db.commit()
                    finally:
                        await self.redis_lock.arelease_lock()

        await self.check_and_update_domain_status()
        logger.info(f"[CRAWL_CONTENT] delete_content end, url_dict:{url_dict}, delete_row_table:{delete_raw_table}")

    async def update_content(self, url_dict):
        logger.info(f"[CRAWL_CONTENT] update_content begin, url_dict:{url_dict}")
        #await self.delete_content(url_dict, False)
        #await self.add_content(url_dict)

        #await self.check_and_update_domain_status()
        
        # Just copy `add_content`
        await self.add_content(url_dict)
        logger.info(f"[CRAWL_CONTENT] update_content end, url_dict:{url_dict}")

    async def check_and_update_domain_status(self):
        logger.info(f"[CRAWL_CONTENT] check_and_update_domain_status")
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            # Step 1: Check current domain_status for the domain
            cursor = await db.execute("SELECT domain_status FROM t_domain_tab WHERE domain = ?", (self.domain,))
            row = await cursor.fetchone()
            if row and row[0] != 4:
                # Step 2: Check if all URLs for the domain have doc_status >= 4
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM t_raw_tab WHERE domain = ? AND doc_status < 4", (self.domain,))
                count_row = await cursor.fetchone()
                if count_row[0] == 0:  # If no records have doc_status < 4
                    timestamp = int(time.time())
                    if await self.redis_lock.aacquire_lock():
                        try:
                            # Step 3: Update domain_status to 4 in t_domain_tab
                            await db.execute(
                                "UPDATE t_domain_tab SET domain_status = ?, mtime = ? WHERE domain = ?", (4, timestamp, self.domain))
                            await db.commit()
                        finally:
                            await self.redis_lock.arelease_lock()
                    logger.info(f"[CRAWL_CONTENT] check_and_update_domain_status, Domain status updated to 4 for domain: {self.domain}")

