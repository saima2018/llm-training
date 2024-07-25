"""该脚本可实现多协程爬取指定url页面中的元素并保存到本地csv文件中。使用时需要根据具体网页调整"""

import asyncio
import logging
import random
import tracemalloc

import structlog
from arsenic import browsers, get_session, services

tracemalloc.start()
import csv
import time

from tqdm import tqdm


class WebParser:
    """
    多协程方式爬取指定url，需要指定页面的css selector和输出保存的文件路径
    """

    def __init__(self, url, ID_source="../data/掌上高考schoolID.json"):
        """
        :param ID_source: 待爬取元素的index,（适用于掌上高考网）
        """
        self.url = url
        with open(ID_source, encoding="utf-8") as f:
            schools_list = eval(f.read())
            self.school_ids = {}
            for school in schools_list:
                name = school["name"]
                school_id = school["school_id"]
                self.school_ids[name] = school_id

    async def scrape_and_save(self, limit, css_selector, output_file):
        service = services.Chromedriver(binary="/usr/bin/chromedriver")
        browser = browsers.Chrome()
        browser.capabilities = {
            "goog:chromeOptions": {
                "args": [
                    "--headless",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ]
            }
        }
        async with limit:
            async with get_session(service, browser) as session:
                time.sleep(random.randint(1, 4))
                await session.get(self.url)
                elements = await session.get_elements(css_selector)
                file = open(output_file, "a", encoding="utf-8")
                csvwriter = csv.writer(file)
                for element in elements:
                    tt = await element.get_text()
                    tt = tt.split("http")
                    csvwriter.writerow([tt[0]])
                    for t in tt[1:]:
                        csvwriter.writerow(["http" + t])

                file.close()

    async def run_scraper_in_loop(self):
        loop = asyncio.get_event_loop()
        limit = asyncio.Semaphore(5)

        retry = 1
        target = 856
        while retry:
            try:
                tasks = []
                counter = 0
                for name, id in self.school_ids.items():
                    counter += 1
                    if counter > target:
                        tasks.append(
                            loop.create_task(
                                self.scrape_and_save(self.url, limit, name)
                            )
                        )

                print("len tasks: ", len(tasks))

                iter = 0
                for t in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                    iter += 1
                    await t

            except:
                print("target, iter, counter before: ", target, iter, counter)
                target = target + iter - 1
                print("target after: ", target)


def set_arsenic_log_level(level=logging.DEBUG):
    logger = logging.getLogger("arsenic")

    def logger_factory():
        return logger

    structlog.configure(logger_factory=logger_factory)
    logger.setLevel(level)


if __name__ == "__main__":
    set_arsenic_log_level()
    zsgk = WebParser()
    asyncio.run(zsgk.run_scraper_in_loop())
