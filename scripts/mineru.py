import logging
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv
from pymongo import MongoClient, errors
from requests.adapters import HTTPAdapter
from tqdm import tqdm

load_dotenv()


# --- 配置 ---
@dataclass
class Config:
    API_HOST: str = os.getenv("API_HOST", "http://localhost")
    API_PORT: str = os.getenv("API_PORT", "8000")
    BACKEND: str = os.getenv("MINERU_BACKEND", "hybrid-auto-engine")

    MONGO_HOST: str = os.getenv("MONGO_HOST", "localhost")
    MONGO_PORT: str = os.getenv("MONGO_PORT", "27017")
    MONGO_USERNAME: str = os.getenv("MONGO_USERNAME", "")
    MONGO_PASSWORD: str = os.getenv("MONGO_PASSWORD", "")
    MONGO_DB: str = os.getenv("MONGO_DB", "literature")
    MONGO_COLLECTION: str = os.getenv("MONGO_COLLECTION", "literature_cmp")

    PAPER_BASE_DIR: Path = Path(os.getenv("PAPER_BASE_DIR", "/data/work/cmpdc/papers"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1"))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "256"))
    TIMEOUT: tuple = (120, 3700)
    TASK_POLL_INTERVAL_SECONDS: float = float(
        os.getenv("TASK_POLL_INTERVAL_SECONDS", "5")
    )
    TASK_TIMEOUT_SECONDS: float = float(os.getenv("TASK_TIMEOUT_SECONDS", "7200"))

    @property
    def base_url(self) -> str:
        return f"{self.API_HOST}:{self.API_PORT}"

    @property
    def tasks_url(self) -> str:
        return f"{self.base_url}/tasks"

    @property
    def mongo_uri(self) -> str:
        auth = (
            f"{self.MONGO_USERNAME}:{self.MONGO_PASSWORD}@"
            if self.MONGO_USERNAME and self.MONGO_PASSWORD
            else ""
        )
        return f"mongodb://{auth}{self.MONGO_HOST}:{self.MONGO_PORT}/?directConnection=true"


cfg = Config()
VISUAL_TYPES = {"image", "table", "chart"}


# --- 日志设置 ---
class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging():
    logger = logging.getLogger("mineru")
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    if logger.handlers:
        return logger

    # 控制台日志 (tqdm 兼容)
    console_handler = TqdmHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # 文件日志 (错误日志)
    file_handler = logging.FileHandler("error.log", encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()

# --- 数据库 ---
client = MongoClient(cfg.mongo_uri)
db = client[cfg.MONGO_DB]
collection = db.get_collection(cfg.MONGO_COLLECTION)
# 解析结果单独存储，避免大字段影响主表查询性能
parsed_collection = db.get_collection(f"{cfg.MONGO_COLLECTION}_parsed")


def _build_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=cfg.MAX_WORKERS, pool_maxsize=cfg.MAX_WORKERS
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_pdf_path(doi: str):
    prefix, _, suffix = doi.partition("/")
    return cfg.PAPER_BASE_DIR / prefix / f"{suffix.replace('/', '_')}.pdf"


def get_pdf_page_count(pdf_path: Path):
    try:
        with subprocess.Popen(
            ["pdfinfo", str(pdf_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        ) as proc:
            stdout, _ = proc.communicate(timeout=30)
            if proc.returncode == 0:
                match = re.search(rb"Pages:\s+(\d+)", stdout)
                if match:
                    return int(match.group(1))
        return
    except Exception:
        return


def _mark_batch_failed(stem_to_doc_map: dict[str, str], reason: str):
    for doi in set(stem_to_doc_map.values()):
        logger.error(f"{reason}: {doi}")
        collection.update_one({"DOI": doi}, {"$set": {"mineru_parsed": False}})


def _submit_task(session: requests.Session, files_to_send: list[tuple]) -> str:
    resp = session.post(
        cfg.tasks_url,
        files=files_to_send,
        data={
            "lang_list": ["ch"],
            "backend": cfg.BACKEND,
            "parse_method": "auto",
            "formula_enable": "true",
            "table_enable": "true",
            "return_md": "true",
            "return_content_list": "true",
            "return_images": "true",
        },
        timeout=cfg.TIMEOUT,
    )
    resp.raise_for_status()

    task_id = resp.json().get("task_id")
    if not task_id:
        raise RuntimeError(f"Task submission did not return task_id: {resp.text}")
    return task_id


def _wait_task(session: requests.Session, task_id: str) -> dict:
    status_url = f"{cfg.tasks_url}/{task_id}"
    deadline = time.monotonic() + cfg.TASK_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        resp = session.get(status_url, timeout=cfg.TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        status = payload.get("status")

        if status == "completed":
            return payload
        if status == "failed":
            raise RuntimeError(
                payload.get("error") or payload.get("message") or "Task failed"
            )

        time.sleep(cfg.TASK_POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"Timed out waiting for task {task_id}")


def _get_task_result(session: requests.Session, task_id: str) -> dict[str, dict]:
    resp = session.get(f"{cfg.tasks_url}/{task_id}/result", timeout=cfg.TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("results", {})


def _extract_visual_items(content_list: list[dict]) -> list[dict]:
    return [item for item in content_list if item.get("type") in VISUAL_TYPES]


def _send_batch(session: requests.Session, batch_docs: list[dict]):
    files_to_send: list[tuple] = []
    stem_to_doc_map: dict[str, str] = {}

    for doc in batch_docs:
        if not (doi := doc.get("DOI")):
            continue

        pdf_path = get_pdf_path(doi)
        if not pdf_path.exists():
            collection.update_one({"DOI": doi}, {"$set": {"mineru_parsed": False}})
            continue

        if (page_count := get_pdf_page_count(pdf_path)) and (page_count > 60):
            collection.update_one({"DOI": doi}, {"$set": {"mineru_parsed": False}})
            continue

        try:
            files_to_send.append(
                ("files", (pdf_path.name, pdf_path.read_bytes(), "application/pdf"))
            )
            stem_to_doc_map[pdf_path.stem] = doi
        except Exception as e:
            logger.error(f"Error reading {pdf_path.name}: {e}")

    if not files_to_send:
        return len(batch_docs)

    try:
        task_id = _submit_task(session, files_to_send)
        _wait_task(session, task_id)
        results = _get_task_result(session, task_id)

        handled_dois: set[str] = set()
        for filename, res in results.items():
            if (doi := stem_to_doc_map.get(filename)) is None:
                continue

            try:
                if "md_content" not in res:
                    logger.error(f"Missing markdown result: {doi}")
                    collection.update_one(
                        {"DOI": doi}, {"$set": {"mineru_parsed": False}}
                    )
                    handled_dois.add(doi)
                    continue

                content_list = res.get("content_list") or []
                if not isinstance(content_list, list):
                    content_list = []
                parsed_collection.update_one(
                    {"DOI": doi},
                    {
                        "$set": {
                            "DOI": doi,
                            "markdown": res.get("md_content", ""),
                            "content_list": content_list,
                            "visual_items": _extract_visual_items(content_list),
                            "images": res.get("images", {}),
                        }
                    },
                    upsert=True,
                )
                collection.update_one(
                    {"DOI": doi},
                    {"$set": {"mineru_parsed": True}},
                )
                handled_dois.add(doi)
            except errors.DocumentTooLarge:
                logger.error(f"DocumentTooLarge: {doi}")
                collection.update_one({"DOI": doi}, {"$set": {"mineru_parsed": False}})
                handled_dois.add(doi)
            except Exception as e:
                logger.error(f"DB error for {doi}: {e}")

        for doi in set(stem_to_doc_map.values()) - handled_dois:
            logger.error(f"Missing parse result: {doi}")
            collection.update_one({"DOI": doi}, {"$set": {"mineru_parsed": False}})

        return len(batch_docs)

    except Exception as e:
        logger.error(f"Batch error: {e}")
        _mark_batch_failed(stem_to_doc_map, "Batch failed")
        return len(batch_docs)


def main():
    logger.info("MinerU Batch Processing Started")
    logger.info(
        f"API: {cfg.tasks_url} | Batch Size: {cfg.BATCH_SIZE} | Workers: {cfg.MAX_WORKERS}"
    )

    # 只抓取从未处理过的任务（mineru_parsed 字段不存在）
    query = {"mineru_parsed": {"$ne": True}}
    all_docs = list(collection.find(query))

    logger.info(f"Found {len(all_docs)} tasks.")
    batches = [
        all_docs[i : i + cfg.BATCH_SIZE]
        for i in range(0, len(all_docs), cfg.BATCH_SIZE)
    ]

    total_start = time.time()
    session = _build_session()

    try:
        with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
            with tqdm(total=len(all_docs), desc="Processing", unit="file") as pbar:
                # 分块提交，每次只提交 10,000 个任务到队列中，防止内存爆炸和启动卡顿
                for chunk_start in range(0, len(batches), 10000):
                    chunk = batches[chunk_start : chunk_start + 10000]
                    futures = [executor.submit(_send_batch, session, b) for b in chunk]

                    for fut in as_completed(futures):
                        try:
                            pbar.update(fut.result())

                        except Exception as e:
                            logger.error(f"Worker thread error: {e}")
                            pbar.update(1)  # 即使报错也推动进度条
    finally:
        session.close()

    total_time = time.time() - total_start
    logger.info("-" * 30)
    logger.info(f"Total: {len(all_docs)} | Time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
