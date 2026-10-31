# -*- coding: utf-8 -*-
"""
================================================================================
  KnowSearch 知识检索问答平台（单文件 RAG 知识问答系统）
================================================================================
  功能：
    1. Web 界面拖拽/点击上传文档（支持 PDF / TXT / Markdown / DOCX）
    2. 文档智能切片（段落优先 + 滑动窗口兜底，chunk_size=500，overlap=50），
       并向量化存入本地向量数据库 ChromaDB（持久化存储，无需额外部署服务）
    3. 自然语言提问：Top-5 向量召回（余弦相似度）+ 大模型流式生成回答
    4. 回答自动标注引用来源（文档名 + 片段号 + 相似度 + 片段摘录）
    5. 多轮对话上下文连贯，支持一键清空对话历史
    6. 知识库管理：查看文档列表、删除单个文档、清空整个知识库

  技术栈：
    - Python + Gradio（单文件部署，启动即用）
    - ChromaDB（本地持久化向量库，目录为 CHROMA_DIR）
    - sentence-transformers（BAAI/bge-small-zh-v1.5 中文嵌入模型）
    - OpenAI 兼容 API（通过 .env 配置 base_url，
      支持 DeepSeek / 硅基流动 SiliconFlow / Ollama / vLLM 等第三方或本地模型）

  启动步骤：
    1) pip install -r requirements.txt
    2) 复制 .env.example 为 .env，填写 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
    3) python app.py
    4) 浏览器自动打开 http://127.0.0.1:7860

  说明：首次运行会自动联网下载嵌入模型权重（约 90MB），之后可离线使用。
================================================================================
"""
import hashlib
import inspect
import json
import os
import re
import threading
import traceback
import uuid
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import gradio as gr
import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 读取项目根目录下的 .env 文件（文件不存在时忽略，参数走代码默认值）

# ==============================================================================
# 1. 全局配置（均可通过 .env 文件覆盖）
# ==============================================================================
# ---- 大模型（OpenAI 兼容接口）----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")            # API Key
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")  # Base URL
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")   # 模型名称

# ---- 嵌入模型 ----
# 中文语义向量化模型：优先使用本地已下载的模型（离线可用，国内友好），
# 否则回退到 HuggingFace 模型名（首次运行会自动联网下载，建议国内用户先走 ModelScope 下载）。
_LOCAL_EMBED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bge-small-zh-v1.5")
_DEFAULT_EMBED = _LOCAL_EMBED_DIR if os.path.isdir(_LOCAL_EMBED_DIR) else "BAAI/bge-small-zh-v1.5"
EMBED_MODEL = os.getenv("EMBED_MODEL", _DEFAULT_EMBED)      # 中文语义向量化模型
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")             # cpu / cuda

# ---- 向量数据库（本地持久化，无需额外服务）----
# 使用脚本所在目录的绝对路径，避免进程工作目录不同导致数据"丢失"（曾出现在不同 cwd 下各写各的向量库）
CHROMA_DIR = os.getenv("CHROMA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowsearch_docs")  # 集合名称

# ---- 检索与切片参数 ----
TOP_K = int(os.getenv("TOP_K", "5"))                        # 最终召回条数
PER_DOC_LIMIT = int(os.getenv("PER_DOC_LIMIT", "3"))        # 单文档最多召回条数（防止被同一文档刷屏）
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))            # 切片大小（字符数）
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))       # 相邻切片重叠量

SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx"}  # 支持的文件格式

KB_LOCK = threading.Lock()   # 保护 ChromaDB 并发读写（上传 / 删除 / 检索共用同一把锁）
FILE_LOCK = threading.RLock()  # 保护本地 JSON 数据文件（users / 会话 / 上传历史 / 模型配置）读-改-写的原子性

# ---- AI 大模型运行时配置（初始值来自 .env，可在界面「AI 大模型」中修改并立即生效） ----
LLM_CONFIG = {
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
    "model": OPENAI_MODEL,
}

# 配置持久化文件：界面中应用的配置会写入此文件，下次启动自动恢复（优先级高于 .env）
LLM_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_config.json")

# 一键切换用的预置模型档案（OpenAI 兼容接口）
MODEL_PRESETS = {
    "🐋 DeepSeek 官方 · Chat（对话）": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "🐋 DeepSeek 官方 · Reasoner（深度思考）": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-reasoner"},
    "⚡ 硅基流动 · DeepSeek-V3": {"base_url": "https://api.siliconflow.cn/v1", "model": "deepseek-ai/DeepSeek-V3"},
    "⚡ 硅基流动 · Qwen2.5-72B": {"base_url": "https://api.siliconflow.cn/v1", "model": "Qwen/Qwen2.5-72B-Instruct"},
    "🌙 Moonshot · Kimi 8K": {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
    "🧠 智谱 · GLM-4-Flash": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
    "🌐 OpenAI · GPT-4o-mini": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "🏠 Ollama · 本地模型": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b"},
}


def _load_saved_llm_config():
    """启动时从 llm_config.json 恢复上次应用的模型配置（文件不存在 / 损坏时忽略）"""
    try:
        with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in ("api_key", "base_url", "model"):
                if data.get(k):
                    LLM_CONFIG[k] = data[k]
    except Exception:
        pass


def _save_llm_config():
    """把当前生效的模型配置写入 llm_config.json（下次启动自动恢复）"""
    try:
        with FILE_LOCK:
            with open(LLM_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(LLM_CONFIG, f, ensure_ascii=False, indent=2)
    except Exception:
        traceback.print_exc()


_load_saved_llm_config()


def _llm_status_text() -> str:
    """配置状态卡片文案"""
    return f"✅ 当前模型：**{LLM_CONFIG['model'] or '（未配置）'}**　·　接口：{LLM_CONFIG['base_url'] or '（默认）'}"


def _current_preset_name():
    """反查与当前生效配置（接口+模型）匹配的预置档案名，用于启动时在下拉框中回显当前模型"""
    for name, p in MODEL_PRESETS.items():
        if p["base_url"] == LLM_CONFIG["base_url"] and p["model"] == LLM_CONFIG["model"]:
            return name
    return None


def apply_llm_config(key: str, url: str, model: str):
    """应用在界面中修改的 AI 大模型配置（立即生效并持久化，无需重启），返回状态文案"""
    key = (key or "").strip()
    model = (model or "").strip()
    if not key:
        gr.Warning("API Key 不能为空")
        return gr.update()
    if not model:
        gr.Warning("模型名称不能为空")
        return gr.update()
    LLM_CONFIG["api_key"] = key
    LLM_CONFIG["base_url"] = (url or "").strip()
    LLM_CONFIG["model"] = model
    _save_llm_config()
    gr.Info(f"AI 大模型配置已应用：{model}（已保存，重启后自动恢复）")
    return _llm_status_text()


def switch_model_preset(name):
    """一键切换预置模型：自动填充并应用接口地址与模型名（沿用当前 API Key）
    返回 (接口地址, 模型名称, 状态文案)"""
    preset = MODEL_PRESETS.get(name)
    if not preset:
        return gr.update(), gr.update(), gr.update()
    LLM_CONFIG["base_url"] = preset["base_url"]
    LLM_CONFIG["model"] = preset["model"]
    _save_llm_config()
    gr.Info(f"已切换到「{name}」（沿用当前 API Key，如服务商不同请先更换 Key）")
    return preset["base_url"], preset["model"], _llm_status_text()


def test_llm_connection(key: str, url: str):
    """用当前输入框中的 API Key / 接口地址测试连通性（拉取模型列表，不发真实对话）"""
    key = (key or "").strip()
    url = (url or "").strip()
    if not key:
        return "❌ 请先填写 API Key 再测试"
    try:
        client = OpenAI(api_key=key, base_url=url or None, timeout=20.0)
        data = client.models.list()
        names = [m.id for m in (data.data or [])][:5]
        gr.Info("连接测试成功")
        extra = f"　可用模型（前 5 个）：{', '.join(names)}" if names else ""
        return f"✅ 连接成功！接口与 API Key 有效。{extra}"
    except Exception as e:
        gr.Warning("连接测试失败")
        return f"❌ 连接失败：{str(e)[:200]}"

# ==============================================================================
# 2. 自定义嵌入函数：让 ChromaDB 直接使用 sentence-transformers 中文模型
# ==============================================================================
class BGEEmbeddingFunction(EmbeddingFunction):
    """将 sentence-transformers 的 bge 中文模型封装为 ChromaDB 嵌入函数。

    BAAI/bge-small-zh-v1.5 输出 512 维向量，中文语义效果好、模型小（约 90MB），
    支持以 normalize_embeddings=True 输出单位向量，配合余弦相似度检索。
    """

    def __init__(self, model_name: str = EMBED_MODEL, device: str = EMBED_DEVICE):
        from sentence_transformers import SentenceTransformer
        print(f"正在加载嵌入模型：{model_name}（device={device}）...")
        self._st = SentenceTransformer(model_name, device=device)

    def __call__(self, input: Documents) -> Embeddings:
        """ChromaDB 调用约定：输入文档列表，返回向量列表"""
        texts = [t if t else " " for t in input]
        vecs = self._st.encode(
            texts,
            normalize_embeddings=True,   # 归一化，使内积等价于余弦相似度
            show_progress_bar=False,
        )
        return vecs.tolist()


# ==============================================================================
# 3. 初始化 ChromaDB（本地持久化，无需额外部署服务）
# ==============================================================================
_embed_fn = BGEEmbeddingFunction()  # 嵌入函数全局复用（避免重复加载模型）
_client = chromadb.PersistentClient(path=CHROMA_DIR)

_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embed_fn,
    metadata={"hnsw:space": "cosine"},  # 使用余弦相似度进行向量检索
)


# ==============================================================================
# 4. 文档解析（PDF / TXT / Markdown / DOCX -> 纯文本）
# ==============================================================================
def read_text_with_encoding(path: Path) -> str:
    """读取纯文本文件，自动探测 UTF-8 / GBK（GB18030）编码"""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")  # 兜底：非法字节用替换符


def parse_document(file_path: str, orig_name: str) -> str:
    """按扩展名解析文档为纯文本，返回非空文本"""
    ext = Path(orig_name).suffix.lower()

    if ext == ".pdf":
        # pypdf 提取文本（仅支持文本型 PDF）
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(pages)

    elif ext == ".docx":
        # python-docx：段落 + 表格文本
        from docx import Document
        doc = Document(file_path)
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        text = "\n".join(parts)

    elif ext in (".txt", ".md", ".markdown"):
        text = read_text_with_encoding(Path(file_path))

    else:
        raise ValueError(f"不支持的文件格式：{ext}（仅支持 PDF / TXT / Markdown / DOCX）")

    text = text.strip()
    if not text:
        raise ValueError("文档内容为空，或未能提取到文本（如扫描图片型 PDF），无法入库")
    return text


# ==============================================================================
# 5. 文本切片：段落优先 + 滑动窗口兜底（chunk_size=500，overlap=50）
# ==============================================================================
def split_long_chunk(text: str) -> list:
    """滑动窗口兜底切分：处理单个段落仍超过 chunk_size 的情况"""
    if not text:
        return []  # 空文本直接返回（防御性处理）
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        piece = text[start:end]
        # 尽量在句号 / 叹号 / 问号 / 分号 / 换行等自然边界处断开，提升片段语义完整性
        if end < len(text):
            for sep in ("。", "！", "？", "；", "\n"):
                idx = piece.rfind(sep)
                if idx >= CHUNK_SIZE // 2:  # 边界不能太靠前，否则浪费检索窗口
                    end = start + idx + 1
                    piece = text[start:end]
                    break
        if end <= start:  # 兜底保护：保证每次循环都能前进
            end = start + 1
            piece = text[start:end]
        chunks.append(piece.strip())
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP if (end - CHUNK_OVERLAP) > start else start + 1
    return [c for c in chunks if c.strip()]


def chunk_text(text: str) -> list:
    """混合切片策略：
    1) 段落优先：按空行拆段，把相邻小段落合并到接近 chunk_size（保持段落完整语义）；
    2) 滑动窗口兜底：对单个超长段落用 overlap=50 的滑动窗口切分。
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged = []
    cur = ""
    for para in paragraphs:
        if not cur:
            cur = para
        elif len(cur) + len(para) + 1 <= CHUNK_SIZE:
            cur = cur + "\n" + para  # 相邻小段落合并
        else:
            merged.append(cur)
            cur = para
    if cur:
        merged.append(cur)

    chunks = []
    for blk in merged:
        if len(blk) <= CHUNK_SIZE:
            chunks.append(blk)
        else:
            chunks.extend(split_long_chunk(blk))
    return chunks


# ==============================================================================
# 6. 知识库操作（列表 / 上传 / 删除 / 清空 / 检索）
# ==============================================================================
CLEAR_KB_BTN_TEXT = "🗑️ 清空知识库"            # 清空按钮默认文案
CLEAR_KB_CONFIRM_TEXT = "⚠️ 再次点击确认清空"    # 二次确认状态的按钮文案


def human_size(num_bytes: int) -> str:
    """字节数 -> 人类可读大小（B / KB / MB / GB）"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0


def _user_ids(username: str) -> list:
    """返回当前用户在向量库中的全部片段 id（需在 KB_LOCK 内调用）"""
    return list(((_collection.get(where={"user": username}, include=[]) or {}).get("ids")) or [])


def list_documents(username: str) -> list:
    """从向量库元数据中还原「当前用户」的文档列表（按文档名聚合），含大小 / 片段数 / 上传时间"""
    with KB_LOCK:
        data = _collection.get(where={"user": username}, include=["metadatas"])

    stats = {}
    for meta in data.get("metadatas") or []:
        name = meta.get("doc_name", "未知文档")
        s = stats.setdefault(name, {
            "chunks": 0,
            "upload_time": meta.get("upload_time", "-"),
            "file_size": meta.get("file_size", "-"),
        })
        s["chunks"] += 1

    docs = [
        {
            "doc_name": name,
            "file_size": s["file_size"],
            "chunk_count": s["chunks"],
            "upload_time": s["upload_time"],
            "status": "✅ 已入库",
        }
        for name, s in stats.items()
    ]
    docs.sort(key=lambda d: d["upload_time"], reverse=True)  # 最新上传的排在最前
    return docs


def docs_to_rows(docs: list) -> list:
    """文档列表 -> 表格行（用于 gradio Dataframe）"""
    return [
        [d["doc_name"], d["file_size"], d["chunk_count"], d["upload_time"], d["status"]]
        for d in docs
    ]


def _stats_html(doc_count: int, chunk_count: int) -> str:
    """知识库统计条 HTML：文档数 / 片段数 两段并排大数字卡片"""
    return (
        '<div class="ks-stats">'
        f'<div class="ks-stat"><div class="ks-stat-num">{doc_count}</div>'
        '<div class="ks-stat-label">份文档</div></div>'
        '<div class="ks-stat-divider"></div>'
        f'<div class="ks-stat"><div class="ks-stat-num">{chunk_count}</div>'
        '<div class="ks-stat-label">个片段</div></div>'
        '</div>'
    )


def get_panel_data(username: str):
    """聚合当前用户左侧面板数据：(表格行, 删除下拉选项, 统计信息文案)"""
    docs = list_documents(username)
    rows = docs_to_rows(docs)
    choices = [d["doc_name"] for d in docs]
    total_chunks = sum(d["chunk_count"] for d in docs)
    stats = _stats_html(len(docs), total_chunks)
    return rows, choices, stats


def refresh_doc_panel(username: str):
    """上传/删除/清空后统一刷新左侧面板：(表格数据, 删除下拉框更新, 统计信息)"""
    rows, choices, stats = get_panel_data(username)
    return rows, gr.update(choices=choices, value=None), stats


def _kb_panel(msg: str, username: str):
    """知识库操作完成后的统一输出（6 元组）：
    (表格, 下拉框, 统计, 状态提示, 解除清空二次确认, 恢复清空按钮文案)"""
    rows, dd, stats = refresh_doc_panel(username)
    return rows, dd, stats, msg, False, gr.update(value=CLEAR_KB_BTN_TEXT)


def refresh_scope_choices(username: str):
    """知识库变动（上传 / 删除 / 清空）后，同步刷新聊天区「问答范围」下拉框选项并重置为全库"""
    _, choices, _ = get_panel_data(username)
    return gr.update(choices=choices, value=None)


# ---------------- 上传历史（持久化到 upload_history.json，可查历史添加的文档） ----------------
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_history.json")
HISTORY_LIMIT = 200  # 最多保留最近 200 条记录


def _load_history(username: str) -> list:
    """读取当前用户的上传历史记录（文件为 {用户名: [记录]} 结构）"""
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get(username, [])
    except Exception:
        pass
    return []


def _save_history(username: str, records: list):
    """保存当前用户的上传历史（dict 结构，按用户名隔离，不影响其他用户）"""
    try:
        with FILE_LOCK:
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            data[username] = records[:HISTORY_LIMIT]
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        traceback.print_exc()  # 历史写入失败不影响主流程


def _append_history(username: str, doc_name: str, action: str, file_size: str = "-", chunk_count: int = 0):
    """追加一条历史记录（最新在最前），持久化到本地 JSON 文件"""
    with FILE_LOCK:  # 读-改-写整体加锁，避免并发操作互相覆盖
        records = _load_history(username)
        records.insert(0, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doc_name": doc_name,
            "action": action,  # 新入库 / 覆盖更新 / 删除 / 清空知识库
            "file_size": file_size,
            "chunk_count": chunk_count,
        })
        _save_history(username, records)


def history_to_rows(username: str) -> list:
    """上传历史 -> 表格行"""
    return [
        [r.get("time", "-"), r.get("doc_name", "-"), r.get("action", "-"),
         r.get("file_size", "-"), r.get("chunk_count", 0)]
        for r in _load_history(username)
    ]


def refresh_history(username: str):
    """刷新「上传历史」表格"""
    return history_to_rows(username)


def handle_upload(files, username: str, progress=gr.Progress()):
    """上传处理：解析 -> 切片 -> 向量化 -> 入库（仅写入当前用户的私有知识库）
    返回 7 元组：(表格, 下拉框, 统计, 状态提示, 清空确认复位, 清空按钮复位, 上传组件清空)
    """
    username = (username or "").strip()
    if not username:
        return _kb_panel("❌ 请先登录后再上传文档。", "") + (gr.update(),)
    if not files:
        return _kb_panel("ℹ️ 未选择文件，请先上传文档。", username) + (gr.update(),)

    # 同一批次内按文件名去重（重复选择同名文件时只保留最后一份）
    unique = {}
    for f in files:
        name = getattr(f, "orig_name", None) or Path(f.name).name
        unique[name] = f
    files = list(unique.values())

    ok_count, fail_list, total_chunks = 0, [], 0
    total = len(files)

    for idx, f in enumerate(files):
        tmp_path = f.name                      # 上传到本机的临时路径
        orig_name = getattr(f, "orig_name", None) or Path(tmp_path).name  # 原始文件名
        ext = Path(orig_name).suffix.lower()

        if ext not in SUPPORTED_EXTS:
            fail_list.append(f"{orig_name}：不支持的文件格式（仅支持 PDF / TXT / Markdown / DOCX）")
            continue

        try:
            # ① 解析文档为纯文本
            progress((idx + 0.15) / total, desc=f"① 正在解析 {orig_name}（{idx + 1}/{total}）")
            text = parse_document(tmp_path, orig_name)

            # ② 智能切片（段落优先 + 滑动窗口兜底）
            progress((idx + 0.45) / total, desc=f"② 正在切片 {orig_name}（{idx + 1}/{total}）")
            chunks = chunk_text(text)
            if not chunks:
                fail_list.append(f"{orig_name}：切分后无有效片段")
                continue

            # ③ 向量化并写入 ChromaDB
            progress((idx + 0.75) / total, desc=f"③ 正在向量化入库 {orig_name}（{idx + 1}/{total}）")
            doc_id = uuid.uuid4().hex          # 每次上传生成唯一文档 ID
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                size_str = human_size(os.path.getsize(tmp_path))
            except OSError:
                size_str = "-"
            ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
            metas = [
                {
                    "user": username,
                    "doc_name": orig_name,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "upload_time": now,
                    "file_size": size_str,
                }
                for i in range(len(chunks))
            ]
            with KB_LOCK:
                # 判断该用户是否已存在同名文档，用于区分「新入库 / 覆盖更新」
                existed = len((_collection.get(where={"$and": [{"user": username}, {"doc_name": orig_name}]}, include=[]) or {}).get("ids") or [])
                # 同名文档重新上传时先删除旧版本（覆盖式更新），避免知识库出现重复
                _collection.delete(where={"$and": [{"user": username}, {"doc_name": orig_name}]})
                _collection.add(ids=ids, documents=chunks, metadatas=metas)
            _append_history(username, orig_name, "覆盖更新" if existed else "新入库", size_str, len(chunks))
            ok_count += 1
            total_chunks += len(chunks)
        except Exception as e:
            traceback.print_exc()              # 完整堆栈打印到控制台便于排查
            fail_list.append(f"{orig_name}：{e}")

    if ok_count:
        msg = f"✅ 成功入库 **{ok_count}** 个文档，共写入 **{total_chunks}** 个切片。（同名文档自动覆盖旧版本）"
        gr.Info(f"成功入库 {ok_count} 个文档")
    else:
        msg = "⚠️ 没有文档成功入库。"
    if fail_list:
        msg += "\n\n❌ 以下文档处理失败：\n" + "\n".join(f" - {x}" for x in fail_list)
        gr.Warning("部分文档处理失败，请查看状态详情")
    # 上传完成后清空上传组件，便于下次重新选择文件
    return _kb_panel(msg, username) + (gr.update(value=None),)


def select_doc(evt: gr.SelectData, username: str):
    """点击文档表格中的某一行 -> 同步到删除下拉框 + 聊天区「问答范围」
    （之后可删除、查看内容，或直接对该文档定向提问）"""
    rows, choices, _ = get_panel_data(username)
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if isinstance(idx, int) and 0 <= idx < len(rows):
        name = rows[idx][0]
        return (
            gr.update(choices=choices, value=name),
            f"🔍 已选中「{name}」：可删除、查看内容；聊天区「问答范围」已自动切换为该文档，可直接对它提问。",
            gr.update(choices=choices, value=name),
        )
    return gr.update(choices=choices), "", gr.update()


def _merge_overlap_chunks(chunks: list) -> str:
    """按切片顺序还原文档全文：
    - 相邻滑动窗口切片之间去掉重叠前缀（overlap），避免重复内容
    - 段落型切片之间用空行分隔，保持阅读性
    """
    if not chunks:
        return ""
    out = chunks[0]
    max_ov = max(CHUNK_OVERLAP * 2, 16)  # 边界截断可能使实际重叠量略小于 overlap，放宽搜索范围
    for c in chunks[1:]:
        if not c:
            continue
        ov = 0
        limit = min(len(out), len(c), max_ov)
        for k in range(limit, 0, -1):
            if out.endswith(c[:k]):  # 找到「上一片结尾 = 当前片开头」的最长重叠
                ov = k
                break
        out += c[ov:] if ov else "\n\n" + c
    return out


def view_document_content(doc_name, username: str):
    """查看文档内容：从当前用户的向量库中取出该文档的全部切片，按序号还原为全文
    返回 (文档信息卡片, 内容文本)"""
    if not doc_name:
        gr.Warning("请先选择要查看的文档（点击文档列表中的行，或在下拉框中选择）")
        return gr.update(), gr.update()
    with KB_LOCK:
        data = _collection.get(where={"$and": [{"user": username}, {"doc_name": doc_name}]}, include=["documents", "metadatas"])
    metas = data.get("metadatas") or []
    docs = data.get("documents") or []
    if not docs:
        gr.Warning(f"文档「{doc_name}」不存在或已被删除")
        return gr.update(), gr.update()
    pairs = sorted(zip(metas, docs), key=lambda p: (p[0] or {}).get("chunk_index", 0))
    content = _merge_overlap_chunks([(d or "").strip() for _, d in pairs])
    # 文档信息卡片：名称 / 大小 / 片段数 / 上传时间
    info = next((d for d in list_documents(username) if d["doc_name"] == doc_name), {})
    meta = (
        f"**📄 {doc_name}**\n\n"
        f"大小 **{info.get('file_size', '-')}** · 片段 **{len(docs)}** 个"
        f" · 上传于 {info.get('upload_time', '-')} · 全文约 **{len(content)}** 字"
    )
    gr.Info(f"已加载文档「{doc_name}」的内容（由 {len(docs)} 个切片还原）")
    return meta, content


def delete_document(doc_name, username: str):
    """删除当前用户的某个文档（按用户名 + 文档名删除全部相关切片）"""
    if not doc_name:
        return _kb_panel("ℹ️ 请先在文档列表中点击要删除的行，或在下拉框中选择文档。", username)
    with KB_LOCK:
        existing = _collection.get(where={"$and": [{"user": username}, {"doc_name": doc_name}]}, include=[])
        n_chunks = len(existing.get("ids") or [])
        if n_chunks > 0:
            _collection.delete(where={"$and": [{"user": username}, {"doc_name": doc_name}]})
    if n_chunks == 0:
        return _kb_panel(f"⚠️ 未找到文档「{doc_name}」，它可能已被删除。", username)
    _append_history(username, doc_name, "删除", "-", n_chunks)
    gr.Info(f"已删除文档：{doc_name}")
    return _kb_panel(f"🗑️ 已删除文档「{doc_name}」（移除 {n_chunks} 个切片）。", username)


def clear_knowledge_base(armed, username: str):
    """清空当前用户的整个知识库（二次确认防误触：仅删除该用户自己的文档）"""
    if not armed:
        rows, dd, stats = refresh_doc_panel(username)
        return (True, gr.update(value=CLEAR_KB_CONFIRM_TEXT), rows, dd, stats,
                "⚠️ 此操作将删除**你的全部文档**且不可恢复！请再次点击按钮确认执行。")
    with KB_LOCK:
        n_before = len(_user_ids(username))  # 记录清空前的片段总数（写入上传历史）
        _collection.delete(where={"user": username})
    _append_history(username, "（全部文档）", "清空知识库", "-", n_before)
    gr.Info("已清空你的知识库")
    rows, dd, stats = refresh_doc_panel(username)
    return False, gr.update(value=CLEAR_KB_BTN_TEXT), rows, dd, stats, "🗑️ 已清空你的知识库。"


def retrieve(question: str, scope_doc: str, username: str) -> list:
    """Top-5 向量召回（仅在当前用户的知识库内检索）：余弦相似度排序 + 去重 + 单文档限流

    - 多召回一批（TOP_K*2）用于去重裁剪，最终精确保留 TOP_K 条
    - 按片段文本去重，避免重复片段干扰生成
    - 限制单文档占比，避免上下文被同一份文档淹没
    - scope_doc：非空时仅召回该文档的切片（对选中文档定向提问）
    """
    scoped = bool(scope_doc)
    where = {"user": username}
    if scoped:
        # Chroma 多字段过滤必须用 $and 组合，平铺多键会报 "Expected where to have exactly one operator"
        where = {"$and": [{"user": username}, {"doc_name": scope_doc}]}
    with KB_LOCK:
        count = len((_collection.get(where=where, include=[]) or {}).get("ids") or [])
        if count == 0:
            return []
        n = min(count, max(TOP_K * 2, 1))  # 请求条数不能超过库内总数（或选中文档切片数）
        res = _collection.query(
            query_texts=[question],
            n_results=n,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

    items = []
    docs = res["documents"][0] or []
    metas = res["metadatas"][0] or []
    dists = res["distances"][0] or []
    seen_texts, per_doc = set(), {}

    for text, meta, dist in zip(docs, metas, dists):
        text = (text or "").strip()
        if not text or text in seen_texts:
            continue                      # ① 自动去重：跳过重复片段
        name = meta.get("doc_name", "未知文档")
        # 定向单文档提问时放宽单文档限流（否则最多只能召回 PER_DOC_LIMIT 条）
        doc_limit = TOP_K if scoped else PER_DOC_LIMIT
        if per_doc.get(name, 0) >= doc_limit:
            continue                      # ② 单文档限流
        seen_texts.add(text)
        per_doc[name] = per_doc.get(name, 0) + 1
        # 余弦空间下 distance = 1 - cosine_similarity
        similarity = round(1.0 - float(dist), 4)
        items.append({
            "doc_name": name,
            "chunk_index": meta.get("chunk_index", 0),
            "text": text,
            "similarity": similarity,
        })
        if len(items) >= TOP_K:
            break
    return items


# ==============================================================================
# 7. 大模型问答：检索增强生成（RAG）+ 流式输出 + 引用来源
# ==============================================================================
def chat_stream(message: str, history, scope_doc: str, username: str):
    """聊天主流程：在「当前用户」的知识库内检索 -> 拼接上下文 -> 流式调用 LLM -> 追加引用来源

    scope_doc：问答范围。非空时仅在指定文档内检索（对选中文档定向提问）
    """
    history = list(history or [])
    message = (message or "").strip()
    scope_doc = (scope_doc or "").strip() or None
    username = (username or "").strip()

    if not message:
        yield history
        return

    # 未登录用户强制拦截（防御性：问答页已隐藏，仍做后端校验保证数据隔离）
    if not username:
        history.append({"role": "assistant", "content": "⚠️ 请先登录后再使用问答功能。"})
        yield history
        return

    # 当前用户知识库为空时的友好提示
    with KB_LOCK:
        count = len(_user_ids(username))
    if count == 0:
        history.append({"role": "user", "content": message})
        history.append({
            "role": "assistant",
            "content": "KnowSearch 提示：你的知识库暂无文档，请先上传文档后再提问",
        })
        yield history
        return

    contexts = retrieve(message, scope_doc, username)
    if not contexts:
        # 库内有文档但检索不到相关内容（极少数情况）/ 定向文档内无相关资料
        history.append({"role": "user", "content": message})
        if scope_doc:
            reply = (
                f"在文档「{scope_doc}」中未检索到与您的问题相关的资料。"
                "请尝试换个问法，或将上方「问答范围」清空以检索整个知识库。"
            )
        else:
            reply = "未检索到与您的问题相关的资料，请尝试换个问法，或上传更多相关文档后重试。"
        history.append({"role": "assistant", "content": reply})
        yield history
        return

    # ---------- 组装多轮对话消息 ----------
    system_prompt = (
        "你是一个严谨的知识问答助手，系统名为 KnowSearch。"
        "你必须严格基于下方『参考资料』中的内容回答用户问题，回答应准确、简洁、有条理。"
        "引用规则：回答中需要说明出处时，请在相应句子末尾用 [1]、[2] 等编号标注，"
        "编号对应『参考资料』的条目顺序。"
        "如果参考资料不足以回答，请明确说明“根据现有资料无法完整回答”，"
        "并给出你从资料中能够确认的部分，不得编造或猜测。"
        "请使用与用户提问相同的语言回答。"
    )
    if scope_doc:
        system_prompt += (
            f" 特别说明：本次提问仅针对文档「{scope_doc}」，"
            "请只依据该文档的资料作答，不要使用其他文档的内容。"
        )
    context_text = "\n\n".join(
        f"[{i + 1}] 来源文档：{c['doc_name']}（片段 {c['chunk_index']}）\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    messages = [{"role": "system", "content": system_prompt}]
    # 最近 6 轮历史对话（此处尚未追加当前轮，history[-12:] 即最近 6 轮完整问答），
    # 保持上下文连贯；当前问题随后单独作为最新一条 user 消息追加
    for h in history[-12:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({
        "role": "user",
        "content": f"用户问题：{message}\n\n========== 参考资料 ==========\n{context_text}",
    })

    # ---------- 占位：用户问题 + 空回答（流式填充） ----------
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    # ---------- 流式调用大模型 ----------
    try:
        # 前置校验：API Key 为空 / 仍是占位符 / 含非 ASCII 字符（如中文）时，
        # 直接给出明确提示，避免 httpx 编码 Authorization 头时抛出难懂的 ascii 错误
        key = LLM_CONFIG["api_key"].strip()
        if not key:
            raise RuntimeError("未配置 OPENAI_API_KEY，请在 .env 文件或「AI 大模型」页面中填写")
        if not key.isascii():
            raise RuntimeError(
                "OPENAI_API_KEY 包含非 ASCII 字符（当前值疑似仍是占位符，如「sk-你的密钥」）。"
                "请在「AI 大模型」页面或 .env 文件中替换为真实的 API Key"
            )
        if "你的密钥" in key or "your" in key.lower() and "key" in key.lower():
            raise RuntimeError("OPENAI_API_KEY 疑似仍是占位符，请替换为真实的 API Key")

        client = OpenAI(api_key=key, base_url=LLM_CONFIG["base_url"] or None)
        stream = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=messages,
            stream=True,          # 流式输出
            temperature=0.3,      # 较低温度：忠于资料、减少编造
        )
        answer = ""
        reasoning = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = (delta.content or "") if delta else ""
            answer += piece
            # 深度思考类模型（如 deepseek-reasoner）会额外输出 reasoning_content，实时累积用于可视化
            reasoning += (getattr(delta, "reasoning_content", None) or "") if delta else ""
            history[-1]["content"] = answer
            yield history

        # ---------- 流式结束后：思考过程 + 引用来源卡片 ----------
        final = answer
        if reasoning.strip():
            final = f"**🧠 思考过程**\n\n{reasoning.strip()}\n\n---\n\n{answer}"
        if contexts:
            footnote = f"\n\n---\n\n> 📎 **引用来源（{len(contexts)} 条）**\n>\n"
            for i, c in enumerate(contexts, 1):
                snippet = c["text"].replace("\n", " ")[:120].strip()
                level = "高" if c["similarity"] >= 0.85 else ("中" if c["similarity"] >= 0.70 else "低")
                footnote += (
                    f"> **{i}.** {c['doc_name']} · 片段 {c['chunk_index']}"
                    f" · 相似度 {c['similarity']:.1%} · {level}\n>\n"
                    f"> {snippet}…\n>\n"
                )
            history[-1]["content"] = final + footnote
            yield history

    except Exception as e:
        traceback.print_exc()
        err = str(e)
        # 将编码类错误（通常是 API Key / Base URL 含中文占位符）翻译成明确提示
        if isinstance(e, UnicodeEncodeError) or "codec can't encode" in err:
            hint = (
                "⚠️ 配置中包含非 ASCII 字符（通常是 .env 里的 OPENAI_API_KEY 仍为占位符"
                "「sk-你的密钥」）。请将其替换为真实的 API Key 后重启应用。"
            )
        else:
            hint = (
                f"⚠️ 调用大模型 API 失败：{err}\n\n"
                "请检查「AI 大模型」页面（或 `.env` 文件）中的 API Key、接口地址、模型名称，"
                "以及网络 / 模型服务是否可达。"
            )
        history[-1]["content"] = hint
        yield history


# ==============================================================================
# 7.5 对话会话持久化（历史对话保存 / 加载 / 删除）
# ==============================================================================
CHAT_SESSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_sessions.json")
SESSION_LIMIT = 50  # 最多保留的历史会话数


def _load_chat_sessions(username: str) -> list:
    """读取当前用户的历史会话列表（文件为 {用户名: [会话]} 结构）"""
    try:
        with open(CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get(username, [])
    except Exception:
        pass
    return []


def _save_chat_sessions(sessions: list, username: str):
    """保存当前用户的历史会话列表（dict 结构，按用户名隔离，超出上限截断）"""
    try:
        with FILE_LOCK:
            try:
                with open(CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            data[username] = sessions[:SESSION_LIMIT]
            with open(CHAT_SESSIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        traceback.print_exc()


def _session_title(messages: list) -> str:
    """会话标题 = 首个用户问题（截断展示）"""
    for m in messages or []:
        if m.get("role") == "user":
            t = (m.get("content") or "").strip().replace("\n", " ")
            return t[:24] or "新对话"
    return "新对话"


def sessions_to_rows(sessions: list) -> list:
    """会话列表 -> 表格行（更新时间 / 标题 / 轮数）"""
    return [
        [s.get("updated", ""), _session_title(s.get("messages", [])),
         sum(1 for m in s.get("messages", []) if m.get("role") == "user")]
        for s in sessions
    ]


def persist_session(history, session_id, username: str):
    """每轮问答结束后把当前对话写入当前用户的会话文件，返回 (表格行, 会话 id)"""
    with FILE_LOCK:  # 读-改-写整体加锁，避免并发发送消息时会话文件互相覆盖
        sessions = _load_chat_sessions(username)
        msgs = [m for m in (history or []) if (m.get("content") or "").strip()]
        if not msgs:
            return sessions_to_rows(sessions), session_id
        if not session_id:
            session_id = uuid.uuid4().hex[:12]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target = next((s for s in sessions if s.get("id") == session_id), None)
        if target is None:
            target = {"id": session_id, "created": now}
            sessions.append(target)
        target["updated"] = now
        target["messages"] = msgs
        sessions.sort(key=lambda s: s.get("updated", ""), reverse=True)
        _save_chat_sessions(sessions, username)
    return sessions_to_rows(sessions), session_id


def start_new_session(username: str):
    """清空对话 = 切换到一个全新会话（旧会话已自动保存在当前用户的历史中）"""
    sessions = _load_chat_sessions(username)
    gr.Info("已清空当前对话并开启新会话，旧对话可在左侧「对话历史」中查看 / 加载")
    return [], uuid.uuid4().hex[:12], sessions_to_rows(sessions)


def select_session(evt: gr.SelectData, username: str):
    """点击会话表格行 -> 记录选中的会话 id"""
    sessions = _load_chat_sessions(username)
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if isinstance(idx, int) and 0 <= idx < len(sessions):
        gr.Info(f"已选中对话：{_session_title(sessions[idx].get('messages', []))}")
        return sessions[idx]["id"]
    return gr.update()


def load_session(session_id, username: str):
    """加载选中的历史会话到聊天区（仅当前用户的会话）"""
    if not session_id:
        gr.Warning("请先在对话历史表格中点击选择一个会话")
        return gr.update(), gr.update()
    for s in _load_chat_sessions(username):
        if s.get("id") == session_id:
            msgs = s.get("messages", [])
            gr.Info(f"已加载对话：{_session_title(msgs)}（{sum(1 for m in msgs if m.get('role') == 'user')} 轮）")
            return msgs, session_id
    gr.Warning("所选会话已不存在")
    return gr.update(), gr.update()


def delete_session(session_id, current_sid, username: str):
    """删除选中的历史会话；若删除的是当前会话则同时切换到新会话"""
    if not session_id:
        gr.Warning("请先在对话历史表格中点击选择一个会话")
        return gr.update(), gr.update(), gr.update(), current_sid
    with FILE_LOCK:  # 读-改-写整体加锁，避免并发操作互相覆盖
        sessions = _load_chat_sessions(username)
        remaining = [s for s in sessions if s.get("id") != session_id]
        if len(remaining) == len(sessions):
            gr.Warning("所选会话已不存在")
            return gr.update(), gr.update(), gr.update(), current_sid
        _save_chat_sessions(remaining, username)
    if session_id == current_sid:
        gr.Info("已删除当前对话，并切换到新会话")
        return sessions_to_rows(remaining), None, [], uuid.uuid4().hex[:12]
    gr.Info("已删除所选对话")
    return sessions_to_rows(remaining), None, gr.update(), current_sid


def all_question_rows(keyword: str = "", username: str = "") -> list:
    """遍历当前用户的全部历史会话，提取所有用户问题（支持关键词过滤），
    返回表格行：[所属对话, 问题]（按会话更新时间倒序，最新在前）"""
    kw = (keyword or "").strip().lower()
    rows = []
    for s in _load_chat_sessions(username):
        title = _session_title(s.get("messages", []))
        for m in s.get("messages", []):
            if (m.get("role") or "") != "user":
                continue
            q = (m.get("content") or "").strip()
            if not q:
                continue
            if kw and kw not in q.lower():
                continue  # 关键词过滤（不区分大小写）
            rows.append([title, q.replace("\n", " ")])
    return rows


def select_question(keyword, evt: gr.SelectData, username: str):
    """点击历史问题表格中的某一行 -> 记录该问题全文
    （按当前过滤关键词重新计算行列表，保证行号与表格内容一致）"""
    rows = all_question_rows(keyword, username)
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if isinstance(idx, int) and 0 <= idx < len(rows):
        gr.Info(f"已选中问题：{rows[idx][1][:30]}")
        return rows[idx][1]
    return gr.update()


def use_selected_question(question):
    """修改并提问：把选中的历史问题填入聊天输入框，修改后发送即可重新提问"""
    if not question:
        gr.Warning("请先在「历史问题」表格中点击选择一个问题")
        return gr.update()
    gr.Info("问题已填入输入框，可直接修改，点击「发送」即可重新提问")
    return question


# ==============================================================================
# 8. Gradio 前端界面（科技蓝配色，简洁现代）
# ==============================================================================
_THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")


def _force_light_css(theme) -> str:
    """操作系统为深色模式时，Gradio 会默认把组件配色切成暗色变量，
    与本项目的浅色科技蓝设计冲突。这里提取浅色主题变量块（:root {...}），
    在 :root .dark 选择器下重新声明一遍，从而锁定整页为浅色主题
    （自定义 CSS 注入在主题 CSS 之后，同选择器后者生效）。"""
    try:
        css = theme._get_theme_css()
        m = re.search(r":root\s*\{([^{}]*--body-background-fill[^{}]*)\}", css, re.S)
        return ":root .dark {" + m.group(1) + "}" if m else ""
    except Exception:
        return ""  # 提取失败时回退默认主题行为，不影响主功能


# 页面交互增强 JS：
# ① 聊天输入框按 Enter 直接发送（Shift+Enter 换行；中文输入法组词期间不误触发）
# ② 下拉选择框：选项展开后再次点击即收起（开关切换）；选项列表 portal 到 body 覆盖显示
#    （不被侧边栏容器裁剪），滚动/缩放实时跟随，滚出可视区自动收起
ENTER_JS = """
() => {
  // ---- 移动端 App 壳管理 ----
  // ① 900px 以下：注入顶层 <style>。Gradio 的 scoped CSS 会把 html/body/.gradio-container
  //    这类顶层选择器改写成永不匹配的形式，必须绕开作用域直插顶层样式，
  //    让登录页 / 模型页保持整页自然滚动。
  // ② 768px 以下且处于问答页：切换为「App 壳」——
  //    锁定整页滚动，容器高度 = 可用视口高度（100vh/dvh 效果，避免地址栏遮挡），
  //    聊天区高度按可视区域动态计算；虚拟键盘弹出（visualViewport 收缩）时自动重新布局，
  //    保证底部输入框与「发送」按钮始终不被键盘遮挡。
  (function ksMobileShell() {
    function injectStyle() {
      if (document.getElementById("ks-mobile-scroll-style")) return;
      var s = document.createElement("style");
      s.id = "ks-mobile-scroll-style";
      s.textContent =
        "@media (max-width: 900px) {" +
        "html, body { height: auto !important; overflow: auto !important; -webkit-overflow-scrolling: touch !important; }" +
        ".gradio-container { height: auto !important; min-height: 100vh !important; overflow: visible !important; }" +
        "}";
      document.head.appendChild(s);
    }
    injectStyle();

    // viewport meta：viewport-fit=cover 启用刘海屏安全区；
    // interactive-widget=resizes-content 让安卓键盘弹出时收缩布局视口（输入框不被顶出屏幕）
    (function ksViewport() {
      if (window.innerWidth > 900) return;
      var meta = document.querySelector('meta[name="viewport"]');
      if (!meta) { meta = document.createElement("meta"); meta.name = "viewport"; document.head.appendChild(meta); }
      var v = meta.getAttribute("content") || "";
      var add = [];
      if (v.indexOf("viewport-fit=cover") === -1) add.push("viewport-fit=cover");
      if (v.indexOf("interactive-widget") === -1) add.push("interactive-widget=resizes-content");
      if (add.length) meta.setAttribute("content", v + (v ? ", " : "") + add.join(", "));
    })();

    function isMobile() { return window.innerWidth <= 768; }
    function mainVisible() {
      var el = document.getElementById("main-row");
      while (el && el !== document.body) {
        if (getComputedStyle(el).display === "none") return false;
        el = el.parentElement;
      }
      return !!el;
    }
    function availH() {
      var vv = window.visualViewport;
      return Math.min(window.innerHeight, vv ? vv.height : window.innerHeight);
    }
    // 聊天区高度 = 输入框顶部 - 聊天区顶部（含间距），随视口 / 内容高度自动伸缩
    function fitChat() {
      var panel = document.getElementById("chat-panel");
      var area = document.getElementById("chat-area");
      if (!panel || !area) return;
      if (!isMobile() || !mainVisible()) { area.style.removeProperty("height"); delete area.dataset.ksH; return; }
      var inputEl = document.getElementById("chat-input");
      var areaTop = area.getBoundingClientRect().top;
      var bottomRef = inputEl ? inputEl.getBoundingClientRect().top : panel.getBoundingClientRect().bottom;
      var h = Math.floor(bottomRef - areaTop - 8);
      if (h >= 150 && String(h) !== area.dataset.ksH) {
        area.dataset.ksH = h;
        area.style.height = h + "px";
      }
    }
    function setShell(on) {
      var cont = document.querySelector(".gradio-container");
      var doc = document.documentElement;
      if (on) {
        var h = availH();
        doc.style.setProperty("overflow", "hidden", "important");
        document.body.style.setProperty("overflow", "hidden", "important");
        if (cont) {
          cont.style.setProperty("height", h + "px", "important");
          cont.style.setProperty("min-height", h + "px", "important");
          cont.style.setProperty("overflow", "hidden", "important");
        }
      } else {
        doc.style.removeProperty("overflow");
        document.body.style.removeProperty("overflow");
        if (cont) {
          cont.style.removeProperty("height");
          cont.style.removeProperty("min-height");
          cont.style.removeProperty("overflow");
        }
      }
    }
    function update() {
      setShell(isMobile() && mainVisible());
      fitChat();
    }
    var _ksTick = false;
    function tick() {
      if (_ksTick) return;
      _ksTick = true;
      requestAnimationFrame(function () { _ksTick = false; update(); });
    }
    window.addEventListener("resize", tick);
    window.addEventListener("orientationchange", tick);
    window.addEventListener("load", tick);
    if (window.visualViewport) window.visualViewport.addEventListener("resize", tick);
    // 登录后切到问答页 / 侧边栏抽屉展开等布局变化
    new MutationObserver(tick).observe(document.body, {
      attributes: true, attributeFilter: ["class", "style"], subtree: true,
    });
    setTimeout(update, 120);
    setTimeout(update, 600);
  })();

  // ---- Enter 发送 ----
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Enter" || e.shiftKey || e.isComposing || e.keyCode === 229) return;
    var t = e.target;
    if (!t || !t.closest || !t.closest("#chat-input")) return;
    e.preventDefault();
    var btn = document.querySelector("#send-btn");
    if (btn) btn.click();
  }, true);

  // ---- 下拉选项列表：portal 到 body 覆盖显示 + 固定定位 + 滚动跟随 ----
  function listFor(root) {
    return document.querySelector('ul.options[data-ks-owner="' + root.id + '"]');
  }
  function positionList(ul) {
    var root = document.getElementById(ul.getAttribute("data-ks-owner"));
    if (!root) return;
    var wrap = root.querySelector(".wrap") || root;
    var r = wrap.getBoundingClientRect();
    var need = Math.min(ul.scrollHeight || 320, 320);
    var spaceBelow = window.innerHeight - r.bottom;
    var below = spaceBelow >= need || spaceBelow >= r.top;  // 智能方向：优先向下，下方不足且上方更宽裕时向上
    ul.style.position = "fixed";
    ul.style.left = r.left + "px";
    ul.style.width = r.width + "px";
    ul.style.zIndex = "2147483000";
    if (below) {
      ul.style.top = (r.bottom + 4) + "px";
      ul.style.bottom = "auto";
      ul.style.maxHeight = Math.max(80, spaceBelow - 10) + "px";
    } else {
      ul.style.top = "auto";
      ul.style.bottom = (window.innerHeight - r.top + 4) + "px";
      ul.style.maxHeight = Math.max(80, r.top - 10) + "px";
    }
  }
  function portalList(ul) {
    if (ul.parentElement === document.body) { positionList(ul); return; }
    var root = ul.closest("#delete-dd, #scope-doc, #preset-dd");
    if (!root) return;
    ul.setAttribute("data-ks-owner", root.id);
    document.body.appendChild(ul);
    positionList(ul);
    requestAnimationFrame(function () { positionList(ul); });  // 布局稳定后按真实高度复测方向
  }
  function closeList(root) {
    var inp = root.querySelector("input");
    if (inp) {
      inp.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
      inp.blur();
    }
    // 模拟「点击组件外部」，触发 Gradio 下拉列表的原生关闭逻辑
    ["pointerdown", "mousedown", "click"].forEach(function (type) {
      document.body.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true }));
    });
  }
  function followOpenLists() {
    var lists = document.querySelectorAll("ul.options[data-ks-owner]");
    for (var i = 0; i < lists.length; i++) {
      var ul = lists[i];
      var root = document.getElementById(ul.getAttribute("data-ks-owner"));
      if (!root) { ul.remove(); continue; }
      var r = (root.querySelector(".wrap") || root).getBoundingClientRect();
      if (r.bottom < 0 || r.top > window.innerHeight) { closeList(root); continue; }  // 滚出可视区收起
      positionList(ul);
    }
  }
  var _ksFollowTick = false;
  function scheduleFollow() {
    if (_ksFollowTick) return;
    _ksFollowTick = true;
    requestAnimationFrame(function () { _ksFollowTick = false; followOpenLists(); });
  }
  window.addEventListener("scroll", scheduleFollow, true);
  window.addEventListener("resize", scheduleFollow);
  // 全局观察：Gradio 插入 ul.options 的瞬间 portal 到 body；选项增删时同步箭头旋转状态
  new MutationObserver(function (muts) {
    var changed = false;
    for (var i = 0; i < muts.length; i++) {
      var added = muts[i].addedNodes;
      for (var j = 0; j < added.length; j++) {
        var n = added[j];
        if (n.nodeType !== 1) continue;
        if (n.tagName === "UL" && n.classList.contains("options")) { portalList(n); changed = true; }
        else if (n.querySelectorAll) {
          var inner = n.querySelectorAll("ul.options");
          for (var k = 0; k < inner.length; k++) { portalList(inner[k]); changed = true; }
        }
      }
      if (muts[i].removedNodes && muts[i].removedNodes.length) changed = true;
    }
    if (!changed) return;
    ["#delete-dd", "#scope-doc", "#preset-dd"].forEach(function (sel) {
      var root = document.querySelector(sel);
      if (root) root.classList.toggle("ks-open", !!listFor(root));
    });
  }).observe(document.body, { childList: true, subtree: true });
  // ---- 图标栏随侧边栏展开加宽（切换 ks-rail-wide 类） ----
  function syncRail() {
    var rail = document.querySelector("#ks-icon-rail");
    var sb = document.querySelector("#ks-sidebar");
    if (!rail || !sb) return;
    // 自身或任一祖先 display:none 时 offsetParent 为 null，据此判断侧边栏是否真实可见
    var shown = !!(sb.offsetParent || sb.getClientRects().length);
    rail.classList.toggle("ks-rail-wide", shown);
  }
  (function watchRail() {
    var sb = document.querySelector("#ks-sidebar");
    if (!sb) return;
    new MutationObserver(syncRail).observe(sb, { attributes: true });
    var row = document.getElementById("main-row");
    if (row) new MutationObserver(syncRail).observe(row, { attributes: true });
    if (sb.parentElement) new MutationObserver(syncRail).observe(sb.parentElement, { attributes: true, childList: true });
    syncRail();
  })();
  function setupDropdown(sel) {
    var root = document.querySelector(sel);
    if (!root) return;
    root.addEventListener("mousedown", function (e) {
      if (e.target.closest && e.target.closest("ul.options")) return;  // 点击候选项本身：放行选中
      if (listFor(root)) {
        // 选项已展开：本次点击收起（拦截默认行为，避免重新聚焦弹开）
        e.preventDefault();
        e.stopPropagation();
        root._ksClosing = true;
        setTimeout(function () { root._ksClosing = false; }, 300);
        closeList(root);
      }
    }, true);
    // 收起瞬间拦截后续 click，防止 Gradio 立刻重新弹出选项
    root.addEventListener("click", function (e) {
      if (root._ksClosing && !(e.target.closest && e.target.closest("ul.options"))) {
        e.preventDefault();
        e.stopPropagation();
      }
    }, true);
  }
  setupDropdown("#delete-dd");
  setupDropdown("#scope-doc");
  setupDropdown("#preset-dd");

  // ---- 折叠面板展开 / 收起平滑动画 ----
  function accContent(acc) {
    var kids = acc.children;
    for (var i = 0; i < kids.length; i++) {
      if (kids[i].tagName !== "BUTTON") return kids[i];  // 内容容器（非头部按钮的子节点）
    }
    return null;
  }
  document.addEventListener("click", function (e) {
    var head = e.target && e.target.closest ? e.target.closest("button.label-wrap") : null;
    if (!head || head._ksAnimating) return;
    var acc = head.closest(".accordion");
    // 仅拦截「收起」动作：先做高度/透明度过渡，再放行给 Gradio 真正折叠
    if (!acc || head.getAttribute("aria-expanded") !== "true") return;
    var content = accContent(acc);
    if (!content) return;
    e.preventDefault();
    e.stopPropagation();
    head._ksAnimating = true;
    content.style.overflow = "hidden";
    content.style.height = content.scrollHeight + "px";
    content.style.opacity = "1";
    content.style.transition = "height 0.25s ease, opacity 0.2s ease";
    requestAnimationFrame(function () {
      content.style.height = "0px";
      content.style.opacity = "0";
    });
    setTimeout(function () {
      content.style.height = "";
      content.style.opacity = "";
      content.style.overflow = "";
      content.style.transition = "";
      head.click();  // 合成点击：_ksAnimating 为真时本处理器放行，由 Gradio 完成折叠
      head._ksAnimating = false;
    }, 260);
  }, true);

  // ---- 登录 / 注册胶囊：点击瞬间前端即完成切换（两表单均已预渲染，零后端往返）----
  (function wireAuthTabs() {
    function ksSetAuthTab(tab) {
      var isLogin = tab === "login";
      // 与 CSS 中 #auth-form-reg{display:none!important} 同级优先级，内联声明可正常覆盖
      function show(el, on) { if (el && el.style) el.style.setProperty("display", on ? "flex" : "none", "important"); }
      show(document.getElementById("auth-form-login"), isLogin);
      show(document.getElementById("auth-form-reg"), !isLogin);
      function paint(btn, on) {
        if (!btn || !btn.classList) return;
        if (on) { btn.classList.add("primary"); btn.classList.remove("secondary"); }
        else { btn.classList.add("secondary"); btn.classList.remove("primary"); }
      }
      paint(document.getElementById("auth-tab-login"), isLogin);
      paint(document.getElementById("auth-tab-reg"), !isLogin);
    }
    function bindAuthTab(id, tab) {
      var el = document.getElementById(id);
      if (el) el.addEventListener("click", function () { ksSetAuthTab(tab); }, true);
    }
    ksSetAuthTab("login");  // 初始归一：登录表单显示、注册表单隐藏
    bindAuthTab("auth-tab-login", "login");
    bindAuthTab("auth-tab-reg", "reg");
    // 一键填入演示账号：同时把胶囊切回「登录」页（填入的字段在登录表单里），
    // 并先在本地填好账号密码（与后端 fill_demo 回调幂等，视觉上零等待）
    var demoBtn = document.getElementById("fill-demo-btn");
    if (demoBtn) demoBtn.addEventListener("click", function () {
      ksSetAuthTab("login");
      try {
        function setVal(sel, v) {
          var el = document.querySelector(sel);
          if (!el) return;
          el.value = v;
          el.dispatchEvent(new Event("input", { bubbles: true }));
        }
        setVal("#auth-form-login textarea", "demo");
        setVal("#auth-form-login input[type='password']", "demo123456");
      } catch (e) { /* 后端兜底回填，忽略本地预填异常 */ }
    }, true);
  })();
}
"""

CUSTOM_CSS = _force_light_css(_THEME) + """
/* ================= 全局：填满视口，禁止页面级滚动 ================= */
html, body { height: 100%; overflow: hidden; }
.gradio-container {
  position: relative !important;
  font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif !important;
  background:
    radial-gradient(circle at 14% 8%, rgba(56, 189, 248, 0.12), transparent 26%),
    radial-gradient(circle at 86% 88%, rgba(26, 115, 232, 0.13), transparent 30%),
    radial-gradient(circle at 70% 16%, rgba(11, 78, 162, 0.07), transparent 24%),
    linear-gradient(160deg, #edf3fc 0%, #f7fafe 45%, #e9f1fd 100%) !important;
  height: 100vh !important;
  max-width: none !important;
  padding: 12px 16px !important;
  overflow: hidden !important;
}
/* 科技感网格纹理（极低透明度，不干扰内容阅读） */
.gradio-container::before {
  content: "" !important;
  position: absolute !important; inset: 0 !important;
  pointer-events: none !important;
  background-image:
    linear-gradient(rgba(26, 115, 232, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(26, 115, 232, 0.035) 1px, transparent 1px) !important;
  background-size: 40px 40px !important;
}
/* Gradio 5.50 实际 DOM 层级：
   .gradio-container > main.app > div.wrap > div.contain > 根 column > #main-row
   逐层赋予「可拉伸 + 可收缩」能力（flex + min-height:0），
   才能把视口高度一路传递到主两栏（flex 子项默认 min-height:auto 会被内容撑破） */
.gradio-container > main { flex: 1 1 auto !important; min-height: 0 !important; padding: 0 !important; }
.gradio-container main > .wrap { flex: 1 1 auto !important; min-height: 0 !important; }
.gradio-container main .wrap > .contain { flex: 1 1 auto !important; min-height: 0 !important; }
.gradio-container .contain > .column { flex: 1 1 auto !important; min-height: 0 !important; }
/* 隐藏 Gradio 默认页脚，节省纵向空间 */
.gradio-container footer { display: none !important; }

/* ================= 主两栏：占满剩余高度 ================= */
#main-row {
  flex: 1 1 auto !important; height: auto !important; min-height: 0 !important;
  align-items: stretch !important; flex-wrap: nowrap !important;
}

/* 左侧边栏：内容超高时仅内部滚动（禁止子项换行，保持单列滚动） */
#ks-sidebar {
  background: #ffffff;
  border: 1px solid rgba(26, 115, 232, 0.20);
  border-radius: 20px;
  min-height: 0 !important;
  overflow-y: auto; overflow-x: hidden;
  flex-wrap: nowrap !important;
  padding: 10px 8px;
  box-shadow: 0 0 0 1px rgba(26, 115, 232, 0.05),
              0 0 20px rgba(26, 115, 232, 0.14),
              0 0 44px rgba(56, 189, 248, 0.10),
              0 14px 34px rgba(10, 58, 125, 0.12);
  animation: ks-sidebar-in 0.28s cubic-bezier(0.22, 1, 0.36, 1);
}
@keyframes ks-sidebar-in {
  from { opacity: 0; transform: translateX(10px); }
  to { opacity: 1; transform: translateX(0); }
}
#ks-sidebar > * { flex: 0 0 auto !important; }
#ks-sidebar::-webkit-scrollbar { width: 6px; }
#ks-sidebar::-webkit-scrollbar-thumb { background: #c3d5ee; border-radius: 3px; }
#ks-sidebar::-webkit-scrollbar-track { background: transparent; }

/* 右侧聊天面板：纵向 flex，聊天框自适应占满剩余高度 */
#chat-panel {
  background: rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  border-radius: 20px;
  min-height: 0 !important;
  display: flex !important; flex-direction: column !important;
  flex-wrap: nowrap !important;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(10, 58, 125, 0.04),
              0 8px 24px rgba(10, 58, 125, 0.07),
              0 24px 56px rgba(10, 58, 125, 0.09);
}
#chat-panel > * { flex: 0 0 auto !important; }
#chat-area {
  flex: 1 1 auto !important; min-height: 160px !important;
  height: auto !important;
  display: flex !important; flex-direction: column !important;
}
#chat-area > * { flex: 1 1 auto !important; min-height: 0 !important; height: auto !important; }
#chat-area [style*="height"] { height: 100% !important; max-height: none !important; }

/* ================= 品牌卡片（微渐变 + 顶部高光 + 发光边框） ================= */
.ks-brand-box {
  background: linear-gradient(180deg, #ffffff, #f3f8ff);
  border: 1px solid #e8f0fb;
  border-radius: 16px; padding: 12px 10px 10px; text-align: center;
  margin-bottom: 8px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9),
              0 0 16px rgba(26, 115, 232, 0.10);
}
.ks-logo-mark {
  width: 42px; height: 42px; margin: 0 auto 8px; border-radius: 12px;
  background: linear-gradient(135deg, #0a3a7d 0%, #1a73e8 60%, #38bdf8 100%);
  color: #ffffff; border: none;
  display: flex; align-items: center; justify-content: center;
  font-size: 23px; font-weight: 800;
  box-shadow: 0 4px 12px rgba(26, 115, 232, 0.30),
              0 0 0 5px rgba(26, 115, 232, 0.10);
}
.ks-brand-name { font-size: 19px; font-weight: 800; letter-spacing: 1.5px; color: #0b4ea2; }
.ks-brand-sub { font-size: 11px; opacity: 0.75; margin-top: 3px; letter-spacing: 2px; color: #7c93ad; }

/* 分区小标题（带渐变分隔线） */
.ks-section-title {
  display: flex; align-items: center; gap: 8px;
  font-size: 13px; font-weight: 700; color: #33507a; margin: 10px 2px 6px;
}
.ks-section-title::after {
  content: ""; flex: 1; height: 1px;
  background:
    linear-gradient(90deg, transparent 0%, rgba(56, 189, 248, 0.75) 50%, transparent 100%) no-repeat,
    linear-gradient(90deg, #d9e5f5, transparent);
  background-position: -60% 0, 0 0;
  background-size: 50% 100%, 100% 100%;
  animation: ks-sheen 3.2s ease-in-out infinite;
}
@keyframes ks-sheen {
  0%, 100% { background-position: -60% 0, 0 0; }
  50% { background-position: 160% 0, 0 0; }
}

/* ================= 侧边图标导航栏（收缩态，始终可见） ================= */
#ks-icon-rail {
  border: none !important;
  background: transparent !important;
  padding: 10px 6px !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 8px !important;
  transition: width 0.25s ease !important;
}
/* 展开态：图标栏加宽 + 文字加大（JS 在侧边栏展开时给图标栏加 ks-rail-wide 类） */
#ks-icon-rail.ks-rail-wide {
  width: 96px !important;
  min-width: 96px !important;
}
#ks-icon-rail.ks-rail-wide button span { font-size: 12.5px !important; }
/* 每个图标按钮：卡片化，SVG 线性图标在上、极小文字在下，柔和弥散阴影 */
#ks-icon-rail button {
  position: relative !important;
  width: 100% !important;
  min-height: 58px !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: flex-end !important;
  border-radius: 16px !important;
  border: 1px solid #e3edfa !important;
  background: #ffffff !important;
  padding: 8px 4px 9px !important;
  margin: 0 !important;
  cursor: pointer !important;
  box-shadow: 0 2px 8px rgba(26, 115, 232, 0.10) !important;
  transition: background 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease, transform 0.18s ease;
}
#ks-icon-rail button span {
  display: block !important;
  white-space: nowrap !important;
  overflow: visible !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  line-height: 1.2 !important;
  color: #5b7394 !important;
  text-align: center !important;
  transition: color 0.22s ease, font-size 0.25s ease;
}
/* SVG 线性图标（mask 上色，颜色跟随文字态） */
#ks-icon-rail button::after {
  content: "";
  position: absolute;
  top: 8px; left: 50%;
  transform: translateX(-50%);
  width: 21px; height: 21px;
  background: #5b7394;
  -webkit-mask: var(--ks-icon) center / contain no-repeat;
  mask: var(--ks-icon) center / contain no-repeat;
  transition: background 0.22s ease, transform 0.18s ease;
}
#nav-docs-btn    { --ks-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpath d='M14 2v6h6'/%3E%3Cpath d='M16 13H8'/%3E%3Cpath d='M16 17H8'/%3E%3Cpath d='M10 9H8'/%3E%3C/svg%3E"); }
#nav-history-btn { --ks-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 7v5l3 2'/%3E%3C/svg%3E"); }
#nav-session-btn { --ks-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z'/%3E%3C/svg%3E"); }
#nav-question-btn{ --ks-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='M21 21l-4.35-4.35'/%3E%3C/svg%3E"); }
/* 左侧蓝色指示条（默认透明，选中显现） */
#ks-icon-rail button::before {
  content: "";
  position: absolute;
  left: 0; top: 50%;
  transform: translateY(-50%);
  width: 3px; height: 22px;
  border-radius: 2px;
  background: transparent;
  transition: background 0.22s ease, box-shadow 0.22s ease;
}
#ks-icon-rail button:hover {
  background: #f0f7ff !important;
  border-color: #e3edfa !important;
  box-shadow: 0 2px 10px rgba(26, 115, 232, 0.08);
  transform: translateY(-1px);
}
#ks-icon-rail button:hover::after {
  background: #33507a;
  transform: translateX(-50%) translateY(-1px);
}
/* 选中态：白色卡片 + 左侧蓝色指示条 + 蓝色图标与文字 */
#ks-icon-rail button.primary {
  background: #ffffff !important;
  border-color: #1a73e8 !important;
  box-shadow: 0 6px 16px rgba(26, 115, 232, 0.22);
}
#ks-icon-rail button.primary::before {
  background: #1a73e8;
  box-shadow: 0 0 8px rgba(26, 115, 232, 0.6);
}
#ks-icon-rail button.primary span { color: #0b4ea2 !important; }
#ks-icon-rail button.primary::after { background: #1a73e8; }

/* 内容面板留白 */
#panel-docs, #panel-history, #panel-session, #panel-question {
  padding: 2px 2px !important;
}

/* ================= 文档管理面板：字体与间距统一 ================= */
#panel-docs { gap: 10px !important; }
/* 分区标题：收紧上下留白，与面板节奏一致 */
#panel-docs .ks-section-title {
  margin: 6px 2px 4px !important;
  font-size: 12.5px !important;
}
/* 面板内组件标签统一字号 / 字重 / 颜色 */
#panel-docs .label-wrap span {
  font-size: 12.5px !important; font-weight: 600 !important; color: #33507a !important;
}
/* 上传区标签采用品牌蓝；中央上传图标同步品牌色 */
#upload-zone .label-wrap span { color: #0b4ea2 !important; }
#upload-zone .icon-button { font-size: 22px !important; color: #1a73e8 !important; }
/* 上传状态提示：轻量说明文字，贴近上传区 */
#upload-status p { font-size: 12px !important; color: #7c93ad !important; margin: 0 !important; }
#clear-kb-btn { font-size: 13px !important; }
#delete-btn { font-size: 12.5px !important; }
#view-content-btn { font-size: 13px !important; }

/* 知识库统计条（分块数据卡：大数字渐变 + 分隔线 + 顶部高光） */
#doc-stats {
  background: linear-gradient(180deg, #ffffff, #eef5ff) !important;
  border: 1px solid #d7e6fb !important;
  border-left: 3px solid #1a73e8 !important;
  border-radius: 12px !important;
  padding: 8px 4px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9), 0 2px 8px rgba(26, 115, 232, 0.08);
}
#doc-stats .ks-stats {
  display: flex; align-items: center; justify-content: center;
}
#doc-stats .ks-stat { flex: 1; text-align: center; }
#doc-stats .ks-stat-num {
  font-size: 20px; font-weight: 800; line-height: 1.1;
  background: linear-gradient(135deg, #0a3a7d, #1a73e8 60%, #38bdf8);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}
#doc-stats .ks-stat-label {
  font-size: 11px; font-weight: 600; color: #5b7394; margin-top: 2px;
}
#doc-stats .ks-stat-divider {
  width: 1px; align-self: center; height: 26px;
  background: linear-gradient(180deg, transparent, #a9c8f2, transparent);
}

/* 上传区（品牌色虚线 + 斜纹底 + 悬停/拖入反馈） */
#upload-zone {
  border: 1.5px dashed #1a73e8 !important; border-radius: 14px !important;
  background: repeating-linear-gradient(45deg, #f8fbff 0 10px, #eef5ff 10px 20px) !important;
  transition: border-color 0.2s ease, background 0.2s ease, box-shadow 0.25s ease, transform 0.18s ease;
}
#upload-zone:hover {
  border-color: #0a3a7d !important;
  background: #f0f7ff !important;
  box-shadow: 0 0 0 5px rgba(26, 115, 232, 0.12);
  transform: translateY(-1px);
}

/* 文档表格（表头着色 + 斑马纹 + 悬停高亮，提示可点选） */
#doc-table table { font-size: 12.5px; }
#doc-table thead th { background: #f0f6ff !important; color: #0b4ea2 !important; font-weight: 700; }
#doc-table tbody tr:nth-child(even) { background: #f8fbff; }
#doc-table tbody tr:hover { background: #eaf3ff !important; cursor: pointer; }

/* 上传历史表格（更紧凑） */
#history-table table { font-size: 12px; }
#history-table thead th { background: #f0f6ff !important; color: #0b4ea2 !important; font-weight: 700; }

/* 对话历史表格（更紧凑，提示可点选） */
#session-table table { font-size: 12px; }
#session-table thead th { background: #f0f6ff !important; color: #0b4ea2 !important; font-weight: 700; }
#session-table tbody tr:hover { background: #eaf3ff !important; cursor: pointer; }

/* 历史问题表格（更紧凑，提示可点选） */
#question-table table { font-size: 12px; }
#question-table thead th { background: #f0f6ff !important; color: #0b4ea2 !important; font-weight: 700; }
#question-table tbody tr:hover { background: #eaf3ff !important; cursor: pointer; }

/* 文档内容预览美化 */
#view-content-btn {
  background: linear-gradient(135deg, #1a73e8, #38bdf8) !important;
  color: #ffffff !important; border: none !important; font-weight: 600;
  box-shadow: 0 2px 8px rgba(26, 115, 232, 0.25);
}
#view-content-btn:hover { box-shadow: 0 4px 14px rgba(26, 115, 232, 0.4); transform: translateY(-1px); }
#doc-meta-card {
  background: linear-gradient(135deg, #eaf3ff, #f7fbff);
  border: 1px solid #cfe3fb; border-radius: 10px;
  padding: 8px 12px; margin: 2px 0 6px;
}
#doc-meta-card p { margin: 2px 0; color: #0b4ea2; font-size: 12.5px; }
#llm-status-card {
  background: linear-gradient(135deg, #eaf3ff, #f7fbff);
  border: 1px solid #cfe3fb; border-radius: 10px;
  padding: 8px 12px; margin: 2px 0 6px;
}
#llm-status-card p { margin: 2px 0; color: #0b4ea2; font-size: 12.5px; }
#doc-content-box textarea {
  font-size: 12.5px; line-height: 1.75; color: #243b53;
  background: #fbfdff; border: 1px solid #dbe8fa !important; border-radius: 10px !important;
}
#doc-content-box textarea::-webkit-scrollbar { width: 6px; }
#doc-content-box textarea::-webkit-scrollbar-thumb { background: #c3d5ee; border-radius: 3px; }

/* ================= 下拉选择框美化（删除文档 / 问答范围） ================= */
#delete-dd .wrap, #scope-doc .wrap {
  background: linear-gradient(135deg, #f6faff, #ffffff) !important;
  border: 1.5px solid #a8c8f0 !important;
  border-radius: 10px !important;
  box-shadow: 0 2px 6px rgba(26, 115, 232, 0.10);
  transition: border-color 0.2s, box-shadow 0.2s;
}
#delete-dd .wrap:hover, #scope-doc .wrap:hover {
  border-color: #1a73e8 !important;
  box-shadow: 0 3px 10px rgba(26, 115, 232, 0.20);
}
#delete-dd .wrap:focus-within, #scope-doc .wrap:focus-within {
  border-color: #1a73e8 !important;
  box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.15);
}
#delete-dd input, #scope-doc input {
  color: #0b4ea2 !important; font-weight: 600; font-size: 13px;
}
#delete-dd svg, #scope-doc svg { color: #1a73e8 !important; }
/* 候选项列表 */
ul.options[data-ks-owner="delete-dd"], ul.options[data-ks-owner="scope-doc"] {
  border: 1px solid #cfe3fb !important; border-radius: 10px !important;
  box-shadow: 0 8px 24px rgba(11, 78, 162, 0.15) !important;
  background: #ffffff !important;
  overflow-x: hidden; overflow-y: auto; padding: 4px !important;
}
ul.options[data-ks-owner="delete-dd"] li, ul.options[data-ks-owner="scope-doc"] li {
  border-radius: 7px !important; color: #33507a !important;
  padding: 7px 10px !important; font-size: 12.5px;
}
ul.options[data-ks-owner="delete-dd"] li:hover, ul.options[data-ks-owner="scope-doc"] li:hover,
ul.options[data-ks-owner="delete-dd"] li[aria-selected="true"], ul.options[data-ks-owner="scope-doc"] li[aria-selected="true"] {
  background: linear-gradient(90deg, #eaf3ff, #f4f9ff) !important;
  color: #0b4ea2 !important; font-weight: 600;
}
/* 点击下拉框：不选中文字、不显示输入光标 */
#delete-dd .wrap, #scope-doc .wrap, #preset-dd .wrap { user-select: none; }
#delete-dd input, #scope-doc input, #preset-dd input { caret-color: transparent; user-select: none; }
/* 一键切换下拉框：与另外两个下拉框同款科技蓝样式 */
#preset-dd .wrap {
  background: linear-gradient(135deg, #f6faff, #ffffff) !important;
  border: 1.5px solid #a8c8f0 !important;
  border-radius: 10px !important;
  box-shadow: 0 2px 6px rgba(26, 115, 232, 0.10);
  transition: border-color 0.2s, box-shadow 0.2s;
}
#preset-dd .wrap:hover { border-color: #1a73e8 !important; box-shadow: 0 3px 10px rgba(26, 115, 232, 0.20); }
#preset-dd input { color: #0b4ea2 !important; font-weight: 600; font-size: 13px; }
#preset-dd svg { color: #1a73e8 !important; width: 18px !important; height: 18px !important; transition: transform 0.25s ease; }
#preset-dd.ks-open svg { transform: rotate(180deg); }
ul.options[data-ks-owner="preset-dd"] {
  border: 1px solid #cfe3fb !important; border-radius: 10px !important;
  box-shadow: 0 8px 24px rgba(11, 78, 162, 0.15) !important;
  background: #ffffff !important;
  overflow-x: hidden; overflow-y: auto; padding: 4px !important; animation: ks-drop-in 0.22s ease;
}
ul.options[data-ks-owner="preset-dd"] li {
  border-radius: 8px !important; color: #33507a !important;
  padding: 8px 12px !important; font-size: 13px;
  transition: background 0.15s ease, color 0.15s ease, transform 0.15s ease;
}
ul.options[data-ks-owner="preset-dd"] li:hover {
  background: linear-gradient(90deg, #eaf3ff, #f4f9ff) !important;
  color: #0b4ea2 !important; font-weight: 600; transform: translateX(3px);
}
/* 当前生效项：蓝色实心高亮，与悬停态区分 */
ul.options[data-ks-owner="preset-dd"] li[aria-selected="true"] {
  background: linear-gradient(90deg, #1a73e8, #3b8ef0) !important;
  color: #ffffff !important; font-weight: 700;
}
/* 服务商分组分隔线：DeepSeek(1-2) | 硅基流动(3-4) | Moonshot(5) | 智谱(6) | OpenAI(7) | Ollama(8) */
ul.options[data-ks-owner="preset-dd"] li:nth-child(3),
ul.options[data-ks-owner="preset-dd"] li:nth-child(5),
ul.options[data-ks-owner="preset-dd"] li:nth-child(6),
ul.options[data-ks-owner="preset-dd"] li:nth-child(7),
ul.options[data-ks-owner="preset-dd"] li:nth-child(8) { margin-top: 7px; }
ul.options[data-ks-owner="preset-dd"] li:nth-child(3)::before,
ul.options[data-ks-owner="preset-dd"] li:nth-child(5)::before,
ul.options[data-ks-owner="preset-dd"] li:nth-child(6)::before,
ul.options[data-ks-owner="preset-dd"] li:nth-child(7)::before,
ul.options[data-ks-owner="preset-dd"] li:nth-child(8)::before {
  content: ""; display: block; height: 1px;
  background: linear-gradient(90deg, transparent, #c9defa 12%, #c9defa 88%, transparent);
  margin: -3px -8px 7px;
}
/* 连接测试结果 */
#llm-test-result p { font-size: 12px; margin: 4px 0 0; color: #33507a; word-break: break-all; }

/* ================= 折叠面板展开 / 收起平滑动画 ================= */
/* 展开：内容容器插入时淡入 + 下滑（收起动画由 ENTER_JS 的高度过渡负责） */
.accordion > *:not(button) {
  animation: ks-acc-in 0.28s ease;
  transform-origin: top center;
}
@keyframes ks-acc-in {
  from { opacity: 0; transform: translateY(-8px); }
  to   { opacity: 1; transform: none; }
}
/* 头部美化：卡片式浅蓝渐变条 + 悬停反馈 */
.accordion button.label-wrap {
  background: linear-gradient(135deg, #f2f8ff, #ffffff);
  border: 1px solid #d8e7fa; border-radius: 10px;
  padding: 7px 12px; margin: 2px 0 4px;
  font-size: 12.5px; color: #0b4ea2; font-weight: 600;
  transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.accordion button.label-wrap:hover {
  background: linear-gradient(135deg, #eaf3ff, #f7fbff);
  border-color: #a8c8f0;
  box-shadow: 0 2px 8px rgba(26, 115, 232, 0.12);
}
.accordion button.label-wrap svg { color: #1a73e8; }
/* 箭头美化：加大 + 展开时旋转 180°（衔接动画） */
#delete-dd svg, #scope-doc svg {
  width: 18px !important; height: 18px !important;
  transition: transform 0.25s ease;
}
#delete-dd.ks-open svg, #scope-doc.ks-open svg { transform: rotate(180deg); }
/* 选项列表弹出动画 */
ul.options[data-ks-owner="delete-dd"], ul.options[data-ks-owner="scope-doc"] { animation: ks-drop-in 0.22s ease; }
@keyframes ks-drop-in {
  from { opacity: 0; transform: translateY(-6px); }
  to   { opacity: 1; transform: translateY(0); }
}
/* 按钮点击反馈动画（按下微缩，松开回弹） */
.gradio-container button {
  transition: transform 0.15s ease, box-shadow 0.2s ease, background 0.2s ease, border-color 0.2s ease;
}
.gradio-container button:active { transform: scale(0.97); }

/* ================= 按钮 ================= */
/* 鼠标悬停时显示小手（指针）样式：所有按钮 / 折叠面板头部 / 三个下拉框（含选项）/ 上传区 / 链接 */
.gradio-container button,
.gradio-container button.label-wrap,
.gradio-container a,
#delete-dd .wrap, #scope-doc .wrap, #preset-dd .wrap,
#delete-dd input, #scope-doc input, #preset-dd input,
ul.options[data-ks-owner="delete-dd"] li, ul.options[data-ks-owner="scope-doc"] li, ul.options[data-ks-owner="preset-dd"] li,
#upload-zone { cursor: pointer !important; }
#delete-btn { background: #fff1f0 !important; color: #cf1322 !important; border: 1px solid #ffa39e !important; }
#delete-btn:hover { background: #ffe7e3 !important; }
#clear-kb-btn {
  background: linear-gradient(135deg, #ef4444 0%, #f87171 100%) !important;
  color: #ffffff !important; border: none !important; font-weight: 700;
  border-radius: 12px !important;
  box-shadow: 0 1px 2px rgba(185, 28, 28, 0.18), 0 6px 16px rgba(239, 68, 68, 0.30);
  transition: transform 0.18s ease, box-shadow 0.22s ease;
  cursor: pointer !important;
}
#clear-kb-btn:hover {
  box-shadow: 0 6px 20px rgba(239, 68, 68, 0.42); transform: translateY(-1px);
}
/* ================= 右侧按钮（发送 / 清空对话：统一高度、圆角与阴影） ================= */
#send-btn {
  background: linear-gradient(135deg, #0a3a7d 0%, #1a73e8 55%, #38bdf8 100%) !important;
  color: #ffffff !important;
  border: none !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  border-radius: 12px !important;
  height: 44px !important;
  box-shadow: 0 2px 6px rgba(26, 115, 232, 0.32), 0 8px 20px rgba(56, 189, 248, 0.22);
  transition: transform 0.18s ease, box-shadow 0.22s ease, filter 0.22s ease;
}
#send-btn:hover {
  box-shadow: 0 4px 10px rgba(26, 115, 232, 0.40), 0 12px 28px rgba(56, 189, 248, 0.32);
  transform: translateY(-1px);
  filter: brightness(1.04);
}
#send-btn:active { transform: translateY(0); }

/* 清空对话：白底 + 品牌蓝描边/文字 */
#clear-chat-btn {
  background: #ffffff !important;
  color: #0b4ea2 !important;
  border: 1px solid #1a73e8 !important;
  border-radius: 12px !important;
  height: 44px !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 6px rgba(26, 115, 232, 0.14);
  transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.22s ease, transform 0.18s ease, color 0.2s ease;
}
#clear-chat-btn:hover {
  background: linear-gradient(180deg, #f7faff, #eef5ff) !important;
  border-color: #0a3a7d !important;
  color: #0a3a7d !important;
  box-shadow: 0 2px 6px rgba(16, 42, 76, 0.06), 0 10px 22px rgba(26, 115, 232, 0.18);
  transform: translateY(-1px);
}
#clear-chat-btn:active { transform: translateY(0); }

/* ================= 右侧聊天区 ================= */
.ks-chat-head {
  background: linear-gradient(90deg, #0a3a7d, #1a73e8 60%, #38bdf8);
  color: #ffffff; padding: 13px 18px; border-radius: 16px;
  font-size: 17px; font-weight: 700; margin-bottom: 10px;
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: 0 1px 2px rgba(10, 58, 125, 0.18),
              0 6px 18px rgba(26, 115, 232, 0.24);
}
.ks-chat-head .ks-tag { font-size: 12px; font-weight: 400; opacity: 0.92; }
.ks-dot {
  width: 9px; height: 9px; border-radius: 50%; background: #8effa9;
  display: inline-block; margin-right: 9px;
  box-shadow: 0 0 8px rgba(142, 255, 169, 0.9);
  animation: ks-pulse 2s ease-in-out infinite;
}
@keyframes ks-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.45; transform: scale(0.8); }
}

#chat-area { border: 1px solid #e4ecf8 !important; border-radius: 14px !important; background: #fcfdff !important; }
/* 对话区空白态（AI/RAG 主题强化） */
.ks-empty { text-align: center !important; padding: 30px 20px !important; }
.ks-empty-icon {
  font-size: 46px !important; line-height: 1 !important;
  display: inline-block !important; padding: 16px !important; margin-bottom: 14px !important;
  background: linear-gradient(135deg, #eaf3ff, #d0e8ff) !important;
  border-radius: 50% !important;
  box-shadow: 0 1px 2px rgba(10, 58, 125, 0.08), 0 8px 22px rgba(26, 115, 232, 0.20) !important;
}
.ks-empty-title { font-size: 17px !important; font-weight: 700 !important; color: #123a6b !important; }
.ks-empty-sub { font-size: 13px !important; color: #7c93ad !important; margin-top: 6px !important; letter-spacing: 0.5px !important; }
#chat-area .message.user {
  background: linear-gradient(135deg, #1a73e8, #38bdf8) !important;
  color: #ffffff !important; border-radius: 14px 14px 4px 14px !important;
  box-shadow: 0 2px 8px rgba(26, 115, 232, 0.22);
}
#chat-area .message.bot, #chat-area .message.assistant {
  background: #f4f8fe !important; border: 1px solid #e3ecfa !important;
  color: #1f2f46 !important; border-radius: 14px 14px 14px 4px !important;
}

/* 引用来源卡片（回答末尾的 blockquote：品牌卡片 + 左侧竖线 + 弥散阴影） */
#chat-area .message.bot blockquote {
  background: linear-gradient(180deg, #f7fbff, #eef5ff) !important;
  border: 1px solid #d7e6fb !important;
  border-left: 3px solid #1a73e8 !important;
  border-radius: 10px !important;
  padding: 10px 12px !important;
  margin: 12px 0 0 !important;
  box-shadow: 0 2px 8px rgba(26, 115, 232, 0.08);
}
#chat-area .message.bot blockquote p {
  margin: 3px 0 !important;
  color: #33507a !important;
  font-size: 13px !important;
  line-height: 1.55 !important;
}
#chat-area .message.bot blockquote strong { color: #0b4ea2 !important; }

#chat-input textarea { border-radius: 12px !important; }

/* 页脚 */
.ks-footer { text-align: center; color: #8ca0bd; font-size: 12px; margin-top: 6px; }
.ks-footer p { margin: 0; }

/* ================= 账号状态 + 侧边栏操作入口 ================= */
#auth-status {
  font-size: 13px !important;
  margin: 6px 2px !important;
  padding: 6px 12px !important;
  color: #0b4ea2 !important;
  text-align: center !important;
  background: #eef5ff !important;
  border: 1px solid #d7e6fb !important;
  border-radius: 999px !important;
  position: relative !important;
}
#auth-status::before {
  content: "" !important;
  position: absolute !important;
  left: 12px; top: 50% !important;
  transform: translateY(-50%) !important;
  width: 8px; height: 8px !important;
  border-radius: 50% !important;
  background: #22c55e !important;
  box-shadow: 0 0 6px rgba(34, 197, 94, 0.8) !important;
}
#auth-status p { margin: 0 !important; }
#auth-actions { gap: 8px !important; margin: 2px 0 4px !important; }
#open-llm-btn, #logout-btn {
  flex: 1 !important; border-radius: 10px !important; border: none !important;
  font-weight: 700 !important; transition: transform 0.15s ease, box-shadow 0.2s ease !important;
  cursor: pointer !important;
}
#open-llm-btn:hover, #logout-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(26, 115, 232, 0.18);
}

/* ================= 独立页面（登录 / AI 模型配置） ================= */
#auth-page, #llm-page {
  position: relative !important;
  flex: 1 1 auto !important; min-height: 0 !important;
  overflow-y: auto !important; padding: 24px !important;
  align-items: center !important;
  background:
    radial-gradient(circle at 18% 22%, rgba(26, 115, 232, 0.10), transparent 32%),
    radial-gradient(circle at 82% 76%, rgba(74, 154, 245, 0.12), transparent 34%),
    linear-gradient(#f5f9ff, #eef5fe) !important;
}
/* 科技感网格背景（登录 / 模型页） */
#auth-page::before, #llm-page::before {
  content: "" !important;
  position: absolute !important; inset: 0 !important;
  background-image:
    linear-gradient(rgba(26, 115, 232, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(26, 115, 232, 0.06) 1px, transparent 1px) !important;
  background-size: 36px 36px !important;
  pointer-events: none !important;
}
.ks-page-card {
  position: relative !important; z-index: 1 !important;
  width: 100% !important; max-width: 560px !important;
  background: #ffffff; border-radius: 20px; padding: 26px 30px;
  border: 1px solid #e8f0fb;
  box-shadow: 0 1px 2px rgba(10, 58, 125, 0.04),
              0 8px 24px rgba(10, 58, 125, 0.08),
              0 24px 56px rgba(10, 58, 125, 0.10);
}
.ks-page-title {
  font-size: 19px !important; font-weight: 800 !important; color: #0b4ea2 !important;
  text-align: center !important; margin: 0 0 16px !important;
}

/* ================= 登录页左右分栏（浅色蓝白统一风） ================= */
#auth-split {
  max-width: 1000px !important; width: 100% !important;
  gap: 16px !important; margin: 0 auto !important;
  display: grid !important; grid-template-columns: 1fr 1fr !important;
  align-items: stretch !important;
}
#auth-split > * { margin-top: 0 !important; }
/* 左右两栏等宽等高 */
#auth-left, #auth-right { flex: 1 1 0% !important; min-width: 0 !important; }
#auth-left > :first-child { height: 100% !important; }
#auth-left .html-container { margin: 0 !important; padding: 0 !important; height: 100% !important; }
#auth-left .html-container .prose { height: 100% !important; }
#auth-left { position: relative !important; z-index: 1 !important; }
.ks-auth-left {
  height: 100% !important; min-height: 560px !important;
  display: flex !important; flex-direction: column !important;
  background: linear-gradient(160deg, #ffffff 0%, #eef5ff 55%, #e3f0ff 100%);
  border: 1px solid #d7e6fb; border-radius: 20px;
  padding: 24px 22px;
  box-shadow: 0 1px 2px rgba(10, 58, 125, 0.04),
              0 12px 28px rgba(10, 58, 125, 0.10),
              0 24px 56px rgba(10, 58, 125, 0.08);
}
.ks-auth-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
.ks-auth-logo {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, #0a3a7d 0%, #1a73e8 60%, #38bdf8 100%);
  color: #fff; font-weight: 800; font-size: 19px;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 4px 12px rgba(26, 115, 232, 0.30);
}
.ks-auth-brand-name { font-size: 15.5px; font-weight: 800; color: #0b4ea2; letter-spacing: 0.3px; }
.ks-auth-title { font-size: 24px; font-weight: 800; color: #123a6b; margin-bottom: 4px; }
.ks-auth-sub { font-size: 12.5px; color: #5b7394; letter-spacing: 0.5px; margin-bottom: 16px; }
.ks-auth-feats { display: flex; flex-direction: column; gap: 8px; }
.ks-auth-feat {
  display: flex; align-items: center; gap: 12px;
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid #dcebfb; border-radius: 13px;
  padding: 9px 12px;
  box-shadow: 0 2px 8px rgba(26, 115, 232, 0.06);
  transition: transform 0.18s ease, box-shadow 0.22s ease;
}
.ks-auth-feat:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(26, 115, 232, 0.12); }
.ks-auth-feat-icon {
  width: 36px; height: 36px; border-radius: 10px; flex: none;
  background: linear-gradient(135deg, #e8f2ff, #cfe5fd);
  display: flex; align-items: center; justify-content: center; font-size: 18px;
}
.ks-auth-feat-title { font-size: 13.5px; font-weight: 700; color: #0b4ea2; line-height: 1.4; }
.ks-auth-feat-desc { font-size: 12px; color: #5b7394; margin-top: 3px; line-height: 1.45; }
/* 左栏：三步上手流程 */
.ks-auth-sec-title {
  font-size: 13px; font-weight: 800; color: #0b4ea2; letter-spacing: 0.5px;
  margin: 16px 0 8px; display: flex; align-items: center; gap: 6px;
}
.ks-auth-sec-title::after {
  content: ""; flex: 1; height: 1px;
  background: linear-gradient(90deg, #cfe3fa, transparent);
}
.ks-auth-steps { display: flex; flex-direction: column; gap: 7px; }
.ks-auth-step {
  display: flex; align-items: center; gap: 11px;
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid #e3effd; border-radius: 12px;
  padding: 8px 12px;
  transition: transform 0.18s ease, box-shadow 0.22s ease;
}
.ks-auth-step:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(26, 115, 232, 0.10); }
.ks-auth-step-num {
  width: 26px; height: 26px; flex: none; border-radius: 8px;
  background: linear-gradient(135deg, #0a3a7d, #1a73e8 70%, #38bdf8 100%);
  color: #fff; font-size: 13px; font-weight: 800;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 3px 8px rgba(26, 115, 232, 0.28);
}
.ks-auth-step-title { font-size: 13px; font-weight: 700; color: #0b4ea2; line-height: 1.35; }
.ks-auth-step-desc { font-size: 11.5px; color: #5b7394; margin-top: 2px; line-height: 1.4; }
/* 左栏：底部信任条 */
.ks-auth-foot {
  margin-top: auto; padding-top: 16px;
  display: flex; flex-direction: column; gap: 4px;
  font-size: 11.5px; color: #6b84a6; line-height: 1.5;
}
.ks-auth-foot-ver { font-size: 11px; color: #93a8c4; letter-spacing: 0.3px; }

/* 登录 / 注册胶囊切换 */
#auth-tab-row { gap: 8px !important; margin-bottom: 8px !important; }
#auth-tab-login, #auth-tab-reg {
  flex: 1 !important; height: 40px !important; border-radius: 999px !important;
  font-weight: 700 !important; font-size: 13.5px !important; border: none !important;
  transition: background 0.22s ease, color 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
#auth-tab-login.primary, #auth-tab-reg.primary {
  background: linear-gradient(135deg, #0a3a7d, #1a73e8 55%, #38bdf8 100%) !important;
  color: #fff !important;
  box-shadow: 0 4px 14px rgba(26, 115, 232, 0.30);
}
#auth-tab-login.secondary, #auth-tab-reg.secondary {
  background: #ffffff !important; color: #0b4ea2 !important;
  border: 1px solid #1a73e8 !important;
}
#auth-tab-login.secondary:hover, #auth-tab-reg.secondary:hover {
  background: #eef5ff !important;
}
/* 注册表单已预渲染（后端 visible=True），CSS 兜底先隐藏，等前端 JS（wireAuthTabs）
   挂载后由胶囊瞬切控制；!important 与 JS setProperty(...,"important") 同级，内联优先 */
#auth-form-reg { display: none !important; }

/* ================= 登录页右栏表单：字体与间距统一 ================= */
#auth-page { justify-content: center !important; }
/* 删除各组件容器（.form）的灰色底板与阴影：全卡片只保留白色卡片外观 */
#auth-page .form {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
#auth-right .ks-page-card {
  margin: 0 !important; min-height: 560px !important;
  padding: 20px 22px !important; gap: 12px !important;
}
#auth-right .ks-page-title { font-size: 15px !important; margin: 0 0 2px !important; }
#auth-right .form { gap: 10px !important; }
#auth-right .label-wrap span,
#auth-right label.container > span:first-child {
  font-size: 13px !important; font-weight: 600 !important; color: #33507a !important;
  background: transparent !important;
}
#auth-right textarea {
  font-size: 13.5px !important; line-height: 1.5 !important;
  border-radius: 10px !important; min-height: 36px !important;
}
#login-btn, #reg-btn { height: 40px !important; font-size: 14px !important; border-radius: 10px !important; }
#auth-msg p, #reg-msg p { font-size: 12.5px !important; margin: 0 !important; }

/* 演示账号提示区（纯文字排版，无灰色底板；与按钮同一基线） */
#auth-demo-row { gap: 8px !important; margin-top: 4px !important; align-items: center !important; }
#auth-demo-row .html-container { margin: 0 !important; }
.ks-demo-hint {
  display: flex; align-items: center; box-sizing: border-box;
  min-height: 34px; font-size: 13px; color: #5b7394; flex: 1;
  padding: 0 2px; gap: 6px;
}
.ks-demo-tag { font-size: 12.5px; color: #33507a; font-weight: 600; }
.ks-demo-user {
  color: #0b4ea2; font-size: 13px; font-weight: 700;
  letter-spacing: 0.2px; font-family: ui-monospace, Consolas, "Courier New", monospace;
}
.ks-demo-sep { color: #9db4d2; font-weight: 600; }
#fill-demo-btn {
  border-radius: 999px !important; border: 1px solid #1a73e8 !important;
  color: #0b4ea2 !important; background: #ffffff !important;
  font-size: 13px !important; font-weight: 600 !important; height: 34px !important;
  transition: background 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease;
}
#fill-demo-btn:hover {
  background: #eef5ff !important;
  box-shadow: 0 4px 12px rgba(26, 115, 232, 0.15);
  transform: translateY(-1px);
}

/* ================= 流式回答等待动画（三点弹跳打字效果） =================
   Gradio 在流式首块到达前会渲染 .message.bot.pending 等待气泡（aria-label="Loading response"），
   内部为三个 .dot 圆点；这里增强为弹跳动画，移动端观感等同原生「正在输入」动效。 */
#chat-area .message.bot.pending .dot {
  animation: ks-typing-bounce 1.2s infinite ease-in-out;
}
#chat-area .message.bot.pending .dot:nth-child(2) { animation-delay: 0.15s; }
#chat-area .message.bot.pending .dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes ks-typing-bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30% { transform: translateY(-5px); opacity: 1; }
}
@media (prefers-reduced-motion: reduce) {
  #chat-area .message.bot.pending .dot { animation: none !important; opacity: 0.65; }
}

/* ================= 移动端 / 平板响应式适配 ================= */
/* 900px 以下：登录页 / 模型页改为单列堆叠 + 整页自然滚动 */
@media (max-width: 900px) {
  html, body { overflow: auto !important; -webkit-overflow-scrolling: touch; }
  .gradio-container {
    height: auto !important; min-height: 100vh !important;
    overflow: visible !important;
    padding-top: 8px !important; padding-bottom: 12px !important;
    padding-left: calc(8px + env(safe-area-inset-left)) !important;
    padding-right: calc(8px + env(safe-area-inset-right)) !important;
  }
  #auth-page, #llm-page { padding: 4px !important; overflow: visible !important; }

  /* 登录页：单列堆叠，表单卡片放最上面，品牌介绍在下方 */
  #auth-split {
    display: flex !important; flex-direction: column !important;
    gap: 12px !important; max-width: 640px !important; width: 100% !important;
  }
  #auth-right { order: 1 !important; min-width: 0 !important; width: 100% !important; }
  #auth-left  { order: 2 !important; min-width: 0 !important; width: 100% !important; }
  .ks-page-card { max-width: 100% !important; padding: 18px 16px !important; }
  #auth-right .ks-page-card { min-height: 0 !important; }
  .ks-auth-left { min-height: 0 !important; height: auto !important; padding: 18px 16px !important; }
  /* 左栏收紧：隐藏「三步上手」，底部说明上移 */
  .ks-auth-steps { display: none !important; }
  .ks-auth-sec-title { display: none !important; }
  .ks-auth-foot { margin-top: 16px !important; padding-top: 12px !important; }
}

/* 768px 以下：主问答页变为「App 壳」布局 ——
   聊天区占满整屏（高度由 JS 按可视视口动态计算），图标导航下沉为底部导航栏，
   侧边栏（用户信息 / 文档统计 / 历史对话）变为全高覆盖式抽屉弹窗，内部可滚动。 */
@media (max-width: 768px) {
  /* 主区改纵向：聊天面板在上吃满剩余空间，图标栏固定在最底部 */
  #main-row {
    flex-wrap: nowrap !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    gap: 0 !important;
    overflow: hidden !important;
    padding-bottom: env(safe-area-inset-bottom) !important;
  }
  /* 底部导航栏：4 个图标按钮横排（触控目标 ≥44px），浮于抽屉之上始终可点，
     点当前图标即可收起抽屉（同页面切换交互） */
  #ks-icon-rail {
    order: 3 !important;
    flex: 0 0 auto !important;
    width: 100% !important; min-width: 0 !important; max-width: 100% !important;
    flex-direction: row !important; flex-wrap: nowrap !important;
    gap: 4px !important; padding: 6px 6px !important;
    position: relative !important; z-index: 105 !important;
    background: #ffffff !important;
    border-top: 1px solid #e3edfa !important;
    box-shadow: 0 -6px 18px rgba(26, 115, 232, 0.08) !important;
  }
  #ks-icon-rail.ks-rail-wide { width: 100% !important; min-width: 0 !important; }
  #ks-icon-rail button {
    flex: 1 1 0 !important; min-width: 44px !important; min-height: 52px !important;
    padding: 9px 2px 8px !important; border-radius: 13px !important;
  }
  #ks-icon-rail button span,
  #ks-icon-rail.ks-rail-wide button span { font-size: 10.5px !important; }
  #ks-icon-rail button::after { width: 19px !important; height: 19px !important; top: 9px !important; }

  /* 侧边栏：全高覆盖式抽屉（底部让出导航栏 + 8px 间距），历史对话等列表以弹窗形式展示 */
  #ks-sidebar {
    position: fixed !important; z-index: 100 !important;
    top: calc(8px + env(safe-area-inset-top)) !important;
    bottom: calc(85px + env(safe-area-inset-bottom)) !important;
    left: 8px !important; right: 8px !important;
    width: auto !important; min-width: 0 !important;
    border-radius: 16px !important;
    overflow-y: auto !important; overflow-x: hidden !important;
    -webkit-overflow-scrolling: touch !important;
  }
  /* 抽屉内所有按钮触控目标 ≥44px（加载对话 / 删除对话等） */
  #ks-sidebar button { min-height: 44px !important; }

  /* 聊天面板：全宽 + 弹性占满剩余高度 */
  #chat-panel {
    order: 1 !important;
    width: 100% !important; min-width: 0 !important; max-width: 100% !important;
    flex: 1 1 auto !important; min-height: 0 !important;
    border-radius: 14px !important;
    overflow: visible !important;
  }
  /* 聊天区高度由 JS 动态计算（min-height 仅作兜底），内部滚动区跟随填满 */
  #chat-area { min-height: 200px !important; }
  #chat-area [style*="height"] { height: 100% !important; max-height: none !important; }
  /* 移动端收紧气泡最大宽度，避免长文贴边 */
  #chat-area .message-row .message { max-width: 86% !important; }
  .ks-chat-head {
    flex-wrap: wrap !important; gap: 4px 10px !important;
    font-size: 15px !important; padding: 10px 14px !important; border-radius: 12px !important;
  }
  /* 发送 / 清空按钮：触控目标 46px；输入框 16px 字号防 iOS 聚焦自动放大 */
  #send-btn, #clear-chat-btn { min-height: 46px !important; font-size: 14px !important; }
  #chat-input textarea { font-size: 16px !important; min-height: 54px !important; }
  /* 底部品牌条隐藏，为聊天区与输入区腾出空间 */
  .ks-footer { display: none !important; }
  #doc-stats .ks-stat-num { font-size: 17px !important; }
}

/* 480px 以下：字号与留白进一步收紧 */
@media (max-width: 480px) {
  .ks-auth-title { font-size: 23px !important; }
  .ks-auth-sub { font-size: 12.5px !important; }
  .ks-auth-feat { padding: 9px 10px !important; }
  .ks-page-title { font-size: 17px !important; }
  /* 演示账号行：小屏允许换行，账号信息占满一行、按钮换到下一行，避免截断 */
  #auth-demo-row { flex-wrap: wrap !important; row-gap: 6px !important; }
  .ks-demo-hint { flex: 1 1 100% !important; min-width: 0 !important;
                  white-space: normal !important; overflow: visible !important; }
  #fill-demo-btn { flex: 1 1 100% !important; white-space: nowrap !important; }
}
"""


# ==============================================================================
# 账号系统：本地注册 / 登录（开放注册，游客可跳过浏览直接使用）
# ==============================================================================
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
_PBKDF2_ITERATIONS = 120_000  # PBKDF2 迭代次数（密码加盐哈希）
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fa5]{2,20}$")  # 2-20 位中文 / 字母 / 数字 / 下划线


def _load_users() -> dict:
    """读取用户表（文件不存在 / 损坏时返回空字典）"""
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_users(users: dict):
    """保存用户表"""
    try:
        with FILE_LOCK:
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception:
        traceback.print_exc()


def _hash_password(password: str, salt: bytes = None) -> tuple:
    """PBKDF2 加盐哈希，返回 (盐 hex, 哈希 hex)"""
    salt = salt if salt is not None else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def register_user(username: str, password: str) -> tuple:
    """注册新账号，返回 (是否成功, 提示信息)"""
    username = (username or "").strip()
    if not _USERNAME_RE.match(username):
        return False, "用户名需为 2-20 位，仅支持中文 / 字母 / 数字 / 下划线"
    if not password or len(password) < 6:
        return False, "密码至少需要 6 位"
    with FILE_LOCK:  # 查重 + 写入整体加锁，避免并发注册同名账号互相覆盖
        users = _load_users()
        if username in users:
            return False, "该用户名已被注册，请换一个"
        salt_hex, hash_hex = _hash_password(password)
        users[username] = {"salt": salt_hex, "hash": hash_hex,
                           "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        _save_users(users)
    return True, "注册成功"


def verify_login(username: str, password: str) -> tuple:
    """校验登录，返回 (是否成功, 提示信息)"""
    username = (username or "").strip()
    if not username or not password:
        return False, "请输入用户名和密码"
    users = _load_users()
    rec = users.get(username)
    if not rec:
        return False, "用户名不存在，请先注册"
    try:
        salt = bytes.fromhex(rec.get("salt", ""))
        expect = rec.get("hash", "")
        _, actual = _hash_password(password, salt)
    except Exception:
        return False, "账号数据异常，请联系管理员"
    if actual != expect:
        return False, "密码错误"
    return True, "登录成功"


def _ensure_demo_account():
    """确保演示账号存在（demo / demo123456），用于登录页一键体验"""
    if "demo" not in _load_users():
        ok, _ = register_user("demo", "demo123456")
        if ok:
            print("  - 已自动创建演示账号：demo / demo123456")


def _nav_view(kind: str):
    """页面切换：仅控制三个页面的可见性 (问答页, 登录页, 模型页)
    kind 取值 qa / auth / llm。"""
    return (
        gr.update(visible=(kind == "qa")),
        gr.update(visible=(kind == "auth")),
        gr.update(visible=(kind == "llm")),
    )


def _stay_nav():
    """登录 / 注册失败时：三个页面保持不动"""
    return tuple(gr.update() for _ in range(3))


_SIDE_NAV_KEYS = ("docs", "history", "session", "question")


def _toggle_nav(key, active, open_state):
    """点击侧边栏图标：再次点击当前图标则收起/展开，点击其它图标则展开并切换面板。

    返回 11 元组，依次对应：
      nav_active, nav_open, sidebar_col(展开区), 4 个内容面板, 4 个图标按钮
    """
    if active == key:
        open_state = not open_state
    else:
        open_state = True
    active = key
    return (
        active,
        open_state,
        gr.update(visible=open_state),
        *[gr.update(visible=(key == k)) for k in _SIDE_NAV_KEYS],
        # 仅在展开时高亮当前面板图标；收起时全部恢复默认样式
        *[gr.update(variant=("primary" if (open_state and key == k) else "secondary")) for k in _SIDE_NAV_KEYS],
    )


def _stay_panel():
    """登录 / 注册失败时：用户私有面板保持不动（16 个空更新，与 PANEL_OUTPUTS 对齐）"""
    return tuple(gr.update() for _ in range(16))


def _user_state_updates(username: str):
    """登录成功后刷新当前用户的全部私有组件，返回 16 元组（与 PANEL_OUTPUTS 对齐）"""
    rows, choices, stats = get_panel_data(username)
    sessions = _load_chat_sessions(username)
    init_sid = sessions[0]["id"] if sessions else uuid.uuid4().hex[:12]
    init_msgs = sessions[0].get("messages", []) if sessions else []
    return (
        rows,                                       # doc_table
        gr.update(choices=choices, value=None),      # delete_dropdown
        stats,                                      # doc_stats
        history_to_rows(username),                  # history_table
        sessions_to_rows(sessions),                 # session_table
        all_question_rows("", username),            # q_table
        gr.update(choices=choices, value=None),      # scope_doc
        init_msgs,                                  # chatbot
        init_sid,                                   # session_state
        None,                                       # sess_selected
        None,                                       # q_selected
        "",                                         # upload_status
        "",                                         # doc_meta
        "",                                         # doc_content_box
        False,                                      # clear_armed
        gr.update(value=CLEAR_KB_BTN_TEXT),         # clear_kb_btn
    )


def do_login(username, password):
    """登录回调：成功则切回问答页、刷新该用户数据并写入登录状态"""
    ok, msg = verify_login(username, password)
    if not ok:
        return (f"❌ {msg}",) + _stay_nav() + (gr.update(), gr.update(), None) + _stay_panel()
    name = (username or "").strip()
    gr.Info(f"欢迎回来，{name}！")
    return ("",) + _nav_view("qa") + (f"👤 **{name}**", gr.update(visible=True), name) + _user_state_updates(name)


def do_register(username, password, password2):
    """注册回调：注册成功后自动登录、刷新数据并返回问答页"""
    username = (username or "").strip()
    if password != password2:
        return ("❌ 两次输入的密码不一致",) + _stay_nav() + (gr.update(), gr.update(), None) + _stay_panel()
    ok, msg = register_user(username, password)
    if not ok:
        return (f"❌ {msg}",) + _stay_nav() + (gr.update(), gr.update(), None) + _stay_panel()
    gr.Info(f"注册成功，已自动登录：{username}")
    return ("",) + _nav_view("qa") + (f"👤 **{username}**", gr.update(visible=True), username) + _user_state_updates(username)


def do_logout():
    """退出登录回调：清空登录态、回到登录页、清空用户私有面板数据"""
    gr.Info("已退出登录")
    return (
        "🔐 未登录", gr.update(visible=False), None,   # auth_status, logout_btn, auth_user
    ) + _nav_view("auth") + _stay_panel()


# ==============================================================================
# Gradio 版本自适应：theme / css / js 参数的归属位置
#   Gradio 5.x     —— 三个参数属于 gr.Blocks()
#   Gradio 6.0     —— 官方将参数移至 demo.launch()，Blocks 不再接收（破坏性变更）
#   Gradio 6.1+    —— Blocks 恢复接收，但已弃用（触发 DeprecationWarning，推荐 launch()）
# 这里按当前运行环境的函数签名自动选择：launch() 支持时优先走 launch()（兼容 6.0+ 且无弃用警告），
# 否则回退 Blocks()，保证 5.x / 6.0 / 6.1+ 全版本均可运行。
# ==============================================================================
_GRADIO_UI_ARGS = dict(theme=_THEME, css=CUSTOM_CSS, js=ENTER_JS)


def _gradio_ui_arg_target() -> str:
    """返回 'launch' 或 'blocks'，表示 theme/css/js 应传入哪个方法。"""
    try:
        launch_params = set(inspect.signature(gr.Blocks.launch).parameters)
        if {"theme", "css", "js"} <= launch_params:
            return "launch"
    except Exception:
        pass
    return "blocks"


GRADIO_UI_TARGET = _gradio_ui_arg_target()
BLOCKS_UI_ARGS = _GRADIO_UI_ARGS if GRADIO_UI_TARGET == "blocks" else {}
LAUNCH_UI_ARGS = _GRADIO_UI_ARGS if GRADIO_UI_TARGET == "launch" else {}


def build_ui():
    """构建整个页面：左侧知识库管理 + 右侧智能问答"""
    # 启动时默认进入登录页：所有用户私有数据先置空，登录成功后再按用户名加载
    rows, choices, stats = [], [], _stats_html(0, 0)
    init_sid = uuid.uuid4().hex[:12]
    init_msgs = []

    with gr.Blocks(
        title="KnowSearch 知识检索问答平台",
        fill_height=True,  # 填满视口高度：整页无滚动条，聊天区自适应拉伸
        **BLOCKS_UI_ARGS,   # theme/css/js：按 Gradio 版本自动选择 Blocks 或 launch()
    ) as demo:

        auth_user = gr.State(None)  # 当前登录用户名（None = 未登录，无法进入问答页）

        with gr.Row(equal_height=True, elem_id="main-row", visible=False) as main_row:
            nav_active = gr.State("docs")    # 当前选中的侧边栏模块
            nav_open = gr.State(False)       # 侧边栏是否展开（默认收缩，仅显示图标）

            # ==================== 图标导航栏（始终可见，收缩态） ====================
            with gr.Column(scale=0, min_width=68, elem_id="ks-icon-rail"):
                nav_docs_btn = gr.Button("文档管理", variant="secondary", size="sm", elem_id="nav-docs-btn")
                nav_history_btn = gr.Button("上传历史", variant="secondary", size="sm", elem_id="nav-history-btn")
                nav_session_btn = gr.Button("对话历史", variant="secondary", size="sm", elem_id="nav-session-btn")
                nav_question_btn = gr.Button("历史问题", variant="secondary", size="sm", elem_id="nav-question-btn")

            # ==================== 展开内容区（默认收起，点击图标展开） ====================
            with gr.Column(scale=2, min_width=320, elem_id="ks-sidebar", visible=False) as sidebar_col:
                gr.HTML("""
                <div class="ks-brand-box">
                  <div class="ks-logo-mark">K</div>
                  <div class="ks-brand-name">KnowSearch</div>
                  <div class="ks-brand-sub">知 识 检 索 问 答 平 台</div>
                </div>
                """)

                auth_status = gr.Markdown("🔐 未登录", elem_id="auth-status", container=False)
                with gr.Row(elem_id="auth-actions"):
                    open_llm_btn = gr.Button("🤖 AI 大模型", size="sm", variant="secondary", elem_id="open-llm-btn")
                    logout_btn = gr.Button("🚪 退出", size="sm", variant="secondary", elem_id="logout-btn", visible=False)

                doc_stats = gr.HTML(stats, elem_id="doc-stats")

                # ---- 面板：文档管理（默认显示） ----
                with gr.Column(visible=True, elem_id="panel-docs") as panel_docs:
                    gr.HTML('<div class="ks-section-title">📤 文档上传</div>')
                    upload_files = gr.File(
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md", ".markdown", ".docx"],
                        label="上传文档（PDF / TXT / Markdown / DOCX）",
                        elem_id="upload-zone",
                        height=150,  # 压缩上传区高度，保证面板内各区块尽量紧凑
                    )
                    upload_status = gr.Markdown("同名文档将自动覆盖旧版本。", elem_id="upload-status", container=False)

                    # 核心操作：清空知识库（二次确认）
                    clear_kb_btn = gr.Button(CLEAR_KB_BTN_TEXT, variant="stop", elem_id="clear-kb-btn")
                    clear_armed = gr.State(False)  # 「清空知识库」二次确认状态

                    gr.HTML('<div class="ks-section-title">📄 文档列表</div>')
                    doc_table = gr.Dataframe(
                        value=rows,
                        headers=["文件名", "大小", "片段数", "上传时间", "状态"],
                        datatype=["str", "str", "number", "str", "str"],
                        interactive=False,
                        wrap=True,
                        label="已上传文档",
                        elem_id="doc-table",
                    )
                    with gr.Row():
                        delete_dropdown = gr.Dropdown(choices=choices, label="删除文档", scale=3, elem_id="delete-dd")
                        delete_btn = gr.Button("🗑️ 删除", variant="secondary", scale=1, elem_id="delete-btn")
                    view_content_btn = gr.Button("📖 查看内容", variant="secondary", elem_id="view-content-btn")
                    gr.HTML('<div class="ks-section-title">📄 内容预览</div>')
                    doc_meta = gr.Markdown("", elem_id="doc-meta-card")
                    doc_content_box = gr.Textbox(
                        value="",
                        label="文档全文",
                        interactive=False,
                        lines=10,
                        show_copy_button=True,
                        elem_id="doc-content-box",
                    )

                # ---- 面板：上传历史 ----
                with gr.Column(visible=False, elem_id="panel-history") as panel_history:
                    history_table = gr.Dataframe(
                        value=[],
                        headers=["时间", "文件名", "操作", "大小", "片段数"],
                        datatype=["str", "str", "str", "str", "number"],
                        interactive=False,
                        wrap=True,
                        label="上传历史",
                        elem_id="history-table",
                    )
                    refresh_history_btn = gr.Button("🔄 刷新历史", variant="secondary", size="sm")

                # ---- 面板：对话历史 ----
                with gr.Column(visible=False, elem_id="panel-session") as panel_session:
                    session_table = gr.Dataframe(
                        value=[],
                        headers=["更新时间", "标题（首问）", "轮数"],
                        datatype=["str", "str", "number"],
                        interactive=False,
                        wrap=True,
                        label="历史对话",
                        elem_id="session-table",
                    )
                    with gr.Row():
                        load_session_btn = gr.Button("📥 加载对话", variant="primary", size="sm", scale=1)
                        delete_session_btn = gr.Button("🗑️ 删除对话", variant="secondary", size="sm", scale=1)

                # ---- 面板：历史问题 ----
                with gr.Column(visible=False, elem_id="panel-question") as panel_question:
                    q_search = gr.Textbox(
                        value="",
                        placeholder="输入关键词，实时过滤所有历史问题…",
                        show_label=False,
                        container=False,
                        lines=1,
                    )
                    q_table = gr.Dataframe(
                        value=[],
                        headers=["所属对话", "问题"],
                        datatype=["str", "str"],
                        interactive=False,
                        wrap=True,
                        label="历史问题",
                        elem_id="question-table",
                    )
                    edit_q_btn = gr.Button("✏️ 修改并提问", variant="primary", size="sm")
                    q_selected = gr.State(None)  # 历史问题表格中选中的问题全文

                # ==================== 右侧主区域：智能问答 ====================
            with gr.Column(scale=5, min_width=560, elem_id="chat-panel"):
                gr.HTML("""
                <div class="ks-chat-head">
                  <span><span class="ks-dot"></span>KnowSearch 智能问答</span>
                  <span class="ks-tag">✦ RAG · 检索增强生成</span>
                </div>
                """)

                chatbot = gr.Chatbot(
                    value=init_msgs,  # 启动时恢复最近一次对话
                    type="messages", height=540, label="对话", show_label=False, elem_id="chat-area",
                    placeholder=(
                        "<div class='ks-empty'>"
                        "<div class='ks-empty-icon'>🤖</div>"
                        "<div class='ks-empty-title'>向你的知识库提问</div>"
                        "<div class='ks-empty-sub'>RAG 检索增强生成 · 基于你的私有文档回答</div>"
                        "</div>"
                    ),
                )
                session_state = gr.State(init_sid)  # 当前会话 id（用于持久化归档）
                sess_selected = gr.State(None)      # 对话历史表格中选中的会话 id

                scope_doc = gr.Dropdown(
                    choices=choices,
                    value=None,
                    label="🎯 问答范围",
                    elem_id="scope-doc",
                )

                msg = gr.Textbox(
                    placeholder="请输入问题…（Enter 发送）",
                    show_label=False,
                    lines=2,
                    container=False,
                    elem_id="chat-input",
                )

                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary", elem_id="send-btn")
                    clear_chat_btn = gr.Button("清空对话", variant="secondary", elem_id="clear-chat-btn")

                gr.HTML('<div class="ks-footer">KnowSearch · 本地向量检索 + 大模型生成 · 由 Gradio 驱动</div>')

        # ==================== 页 2：登录 / 注册（默认页） ====================
        with gr.Column(elem_id="auth-page", visible=True) as auth_page:
            with gr.Row(elem_id="auth-split", equal_height=False):
                # -------- 左栏：产品宣传区 --------
                with gr.Column(scale=10, min_width=320, elem_id="auth-left"):
                    gr.HTML("""
                    <div class="ks-auth-left">
                      <div class="ks-auth-brand">
                        <div class="ks-auth-logo">K</div>
                        <div class="ks-auth-brand-name">KnowSearch</div>
                      </div>
                      <div class="ks-auth-title">知识检索平台</div>
                      <div class="ks-auth-sub">本地 RAG 检索增强 · 基于你的私有文档回答</div>
                      <div class="ks-auth-feats">
                        <div class="ks-auth-feat">
                          <div class="ks-auth-feat-icon">🤖</div>
                          <div class="ks-auth-feat-body">
                            <div class="ks-auth-feat-title">AI 助手双场景</div>
                            <div class="ks-auth-feat-desc">本地知识库问答 + 智能对话</div>
                          </div>
                        </div>
                        <div class="ks-auth-feat">
                          <div class="ks-auth-feat-icon">🔒</div>
                          <div class="ks-auth-feat-body">
                            <div class="ks-auth-feat-title">隐私检测 · 自动脱敏</div>
                            <div class="ks-auth-feat-desc">数据本地处理，隐私不泄露</div>
                          </div>
                        </div>
                        <div class="ks-auth-feat">
                          <div class="ks-auth-feat-icon">📋</div>
                          <div class="ks-auth-feat-body">
                            <div class="ks-auth-feat-title">行为记录 · 风险可视</div>
                            <div class="ks-auth-feat-desc">操作留痕，风险可控</div>
                          </div>
                        </div>
                      </div>
                      <div class="ks-auth-sec-title">🚀 三步上手</div>
                      <div class="ks-auth-steps">
                        <div class="ks-auth-step">
                          <div class="ks-auth-step-num">1</div>
                          <div class="ks-auth-step-body">
                            <div class="ks-auth-step-title">注册账号</div>
                            <div class="ks-auth-step-desc">或使用演示账号，一键快速体验</div>
                          </div>
                        </div>
                        <div class="ks-auth-step">
                          <div class="ks-auth-step-num">2</div>
                          <div class="ks-auth-step-body">
                            <div class="ks-auth-step-title">上传私有文档</div>
                            <div class="ks-auth-step-desc">支持 PDF / Word / TXT / Markdown，自动切片入库</div>
                          </div>
                        </div>
                        <div class="ks-auth-step">
                          <div class="ks-auth-step-num">3</div>
                          <div class="ks-auth-step-body">
                            <div class="ks-auth-step-title">智能问答</div>
                            <div class="ks-auth-step-desc">本地 RAG 检索增强，即问即答随时追问</div>
                          </div>
                        </div>
                      </div>
                      <div class="ks-auth-foot">
                        <div>🛡️ 文档不出本地 · 数据 100% 本地处理 · 操作全程留痕</div>
                        <div class="ks-auth-foot-ver">KnowSearch v1.0 · 本地 RAG 知识检索问答平台</div>
                      </div>
                    </div>
                    """)
                # -------- 右栏：登录 / 注册表单卡片 --------
                with gr.Column(scale=10, min_width=300, elem_id="auth-right"):
                    with gr.Column(elem_classes="ks-page-card"):
                        gr.HTML('<div class="ks-page-title">🔐 欢迎使用</div>')
                        with gr.Row(elem_id="auth-tab-row"):
                            auth_tab_login = gr.Button("登 录", variant="primary", elem_id="auth-tab-login")
                            auth_tab_reg = gr.Button("注 册", variant="secondary", elem_id="auth-tab-reg")
                        with gr.Column(visible=True, elem_id="auth-form-login") as login_form_col:
                            login_user = gr.Textbox(label="用户名", placeholder="请输入用户名")
                            login_pass = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                            login_btn = gr.Button("登 录", variant="primary", elem_id="login-btn")
                            login_msg = gr.Markdown("", elem_id="auth-msg")
                        # 注意：注册表单必须 visible=True 预渲染。Gradio 5.50 对 visible=False 的
                        # Column 采用延迟渲染（不进 DOM），若初始隐藏则首次切到「注册」需等
                        # 后端往返才出现，表现为切换卡顿。这里两个表单都进 DOM，
                        # 由前端 JS（wireAuthTabs + 下方 CSS）控制显示/隐藏，切换零延迟。
                        with gr.Column(visible=True, elem_id="auth-form-reg") as reg_form_col:
                            reg_user = gr.Textbox(label="用户名", placeholder="2-20 位中英文 / 数字 / 下划线")
                            reg_pass = gr.Textbox(label="密码", type="password", placeholder="至少 6 位")
                            reg_pass2 = gr.Textbox(label="确认密码", type="password", placeholder="再次输入密码")
                            reg_btn = gr.Button("注 册 并 登 录", variant="primary", elem_id="reg-btn")
                            reg_msg = gr.Markdown("", elem_id="reg-msg")
                        with gr.Row(elem_id="auth-demo-row"):
                            gr.HTML('<div class="ks-demo-hint"><span class="ks-demo-tag">演示账号</span><b class="ks-demo-user">demo</b><span class="ks-demo-sep">/</span><b class="ks-demo-user">demo123456</b></div>')
                            fill_demo_btn = gr.Button("一键填入", variant="secondary", size="sm", elem_id="fill-demo-btn")

        # ==================== 页 3：AI 大模型 ====================
        with gr.Column(elem_id="llm-page", visible=False) as llm_page:
            with gr.Column(elem_classes="ks-page-card"):
                gr.HTML('<div class="ks-page-title">🤖 AI 大模型</div>')
                llm_status = gr.Markdown(_llm_status_text(), elem_id="llm-status-card")
                preset_dd = gr.Dropdown(
                    choices=list(MODEL_PRESETS.keys()),
                    label="⚡ 一键切换模型",
                    value=_current_preset_name(),
                    elem_id="preset-dd",
                )
                llm_key = gr.Textbox(label="API Key", type="password", value=LLM_CONFIG["api_key"], placeholder="sk-...")
                llm_url = gr.Textbox(label="接口地址 Base URL", value=LLM_CONFIG["base_url"], placeholder="https://api.deepseek.com/v1")
                llm_model = gr.Textbox(label="模型名称", value=LLM_CONFIG["model"], placeholder="deepseek-chat")
                with gr.Row():
                    apply_llm_btn = gr.Button("💾 应用配置", variant="primary", scale=1)
                    test_llm_btn = gr.Button("🔗 测试连接", variant="secondary", scale=1)
                llm_test_result = gr.Markdown("")
                back_qa_btn2 = gr.Button("← 返回智能问答", variant="secondary", size="sm")

        # ==================== 事件绑定 ====================
        # 侧边栏图标导航：点击展开/收起整个侧边栏，并切换对应内容面板 + 图标选中态
        _nav_outputs = [
            nav_active, nav_open, sidebar_col,
            panel_docs, panel_history, panel_session, panel_question,
            nav_docs_btn, nav_history_btn, nav_session_btn, nav_question_btn,
        ]
        for _key, _btn in zip(
            _SIDE_NAV_KEYS,
            [nav_docs_btn, nav_history_btn, nav_session_btn, nav_question_btn],
        ):
            _btn.click(
                partial(_toggle_nav, _key),
                inputs=[nav_active, nav_open],
                outputs=_nav_outputs,
            )
        # 上传：完成后刷新面板、复位清空确认状态并清空上传组件，联动刷新上传历史与问答范围选项
        upload_files.change(
            handle_upload,
            inputs=[upload_files, auth_user],
            outputs=[doc_table, delete_dropdown, doc_stats, upload_status,
                     clear_armed, clear_kb_btn, upload_files],
        ).then(refresh_history, inputs=auth_user, outputs=history_table).then(
            refresh_scope_choices, inputs=auth_user, outputs=scope_doc)
        # 点击文档表格行 -> 同步选中到删除下拉框 + 聊天区「问答范围」（可定向提问）
        doc_table.select(select_doc, inputs=auth_user, outputs=[delete_dropdown, upload_status, scope_doc])
        # 删除单个文档
        delete_btn.click(
            delete_document,
            inputs=[delete_dropdown, auth_user],
            outputs=[doc_table, delete_dropdown, doc_stats, upload_status,
                     clear_armed, clear_kb_btn],
        ).then(refresh_history, inputs=auth_user, outputs=history_table).then(
            refresh_scope_choices, inputs=auth_user, outputs=scope_doc)
        # 清空知识库（二次确认）
        clear_kb_btn.click(
            clear_knowledge_base,
            inputs=[clear_armed, auth_user],
            outputs=[clear_armed, clear_kb_btn, doc_table, delete_dropdown,
                     doc_stats, upload_status],
        ).then(refresh_history, inputs=auth_user, outputs=history_table).then(
            refresh_scope_choices, inputs=auth_user, outputs=scope_doc)
        # 查看选中文档内容（切片还原全文 + 信息卡片）
        view_content_btn.click(
            view_document_content,
            inputs=[delete_dropdown, auth_user],
            outputs=[doc_meta, doc_content_box],
        )
        # 手动刷新上传历史
        refresh_history_btn.click(refresh_history, inputs=auth_user, outputs=history_table)
        # 发送 / 回车（ENTER_JS 触发发送按钮）：流式问答（按「问答范围」检索），结束后清空输入框并归档会话
        send_btn.click(
            chat_stream, inputs=[msg, chatbot, scope_doc, auth_user], outputs=chatbot
        ).then(lambda: "", None, msg).then(
            persist_session, inputs=[chatbot, session_state, auth_user],
            outputs=[session_table, session_state],
        )
        # 清空对话 = 切换到新会话（旧会话保留在对话历史中）
        clear_chat_btn.click(
            start_new_session, inputs=auth_user, outputs=[chatbot, session_state, session_table]
        )
        # 对话历史管理：点行选中 -> 加载 / 删除
        session_table.select(select_session, inputs=auth_user, outputs=sess_selected)
        load_session_btn.click(
            load_session, inputs=[sess_selected, auth_user], outputs=[chatbot, session_state]
        )
        delete_session_btn.click(
            delete_session, inputs=[sess_selected, session_state, auth_user],
            outputs=[session_table, sess_selected, chatbot, session_state],
        )
        # 历史问题：关键词实时过滤 -> 点行选中问题 -> 「修改并提问」填入输入框
        q_search.change(all_question_rows, inputs=[q_search, auth_user], outputs=q_table)
        q_table.select(select_question, inputs=[q_search, auth_user], outputs=q_selected)
        edit_q_btn.click(use_selected_question, inputs=q_selected, outputs=msg)
        # AI 大模型：一键切换预置模型 / 应用自定义配置 / 测试连接
        preset_dd.change(
            switch_model_preset, inputs=preset_dd, outputs=[llm_url, llm_model, llm_status]
        )
        apply_llm_btn.click(
            apply_llm_config, inputs=[llm_key, llm_url, llm_model], outputs=llm_status
        )
        test_llm_btn.click(
            test_llm_connection, inputs=[llm_key, llm_url], outputs=llm_test_result
        )

        # ==================== 页面切换 ====================
        _nav_outputs = [main_row, auth_page, llm_page]
        # 问答页侧边栏「AI 大模型」入口（仅登录后可见，无需额外门禁）
        open_llm_btn.click(lambda: _nav_view("llm"), outputs=_nav_outputs)
        # AI 大模型页「返回智能问答」
        back_qa_btn2.click(lambda: _nav_view("qa"), outputs=_nav_outputs)

        # ==================== 账号：登录 / 注册 / 退出 ====================
        PANEL_OUTPUTS = [doc_table, delete_dropdown, doc_stats, history_table, session_table,
                         q_table, scope_doc, chatbot, session_state, sess_selected, q_selected,
                         upload_status, doc_meta, doc_content_box, clear_armed,
                         clear_kb_btn]
        login_btn.click(
            do_login, inputs=[login_user, login_pass],
            outputs=[login_msg] + _nav_outputs + [auth_status, logout_btn, auth_user] + PANEL_OUTPUTS,
        )
        reg_btn.click(
            do_register, inputs=[reg_user, reg_pass, reg_pass2],
            outputs=[reg_msg] + _nav_outputs + [auth_status, logout_btn, auth_user] + PANEL_OUTPUTS,
        )
        # 登录 / 注册胶囊切换：由前端 JS（wireAuthTabs）瞬切完成，不走后端队列，
        # 避免聊天流式任务占住队列、或部署网络往返导致切换卡顿。
        # 演示账号一键填入（填入的是登录表单字段，前端会同步切回「登录」胶囊）
        fill_demo_btn.click(
            lambda: (gr.update(value="demo"), gr.update(value="demo123456")),
            None, outputs=[login_user, login_pass],
        )
        logout_btn.click(
            do_logout, outputs=[auth_status, logout_btn, auth_user] + _nav_outputs + PANEL_OUTPUTS,
        )

    return demo


_ensure_demo_account()
with warnings.catch_warnings():
    # Gradio 5.x 下 theme/css/js 走 Blocks 会提示「6.0 将移除」；本应用已按版本自适应
    # （6.0+ 自动改走 launch()，见 GRADIO_UI_TARGET），此处提示属预期，静默处理。
    warnings.filterwarnings(
        "ignore",
        message=r"The '(theme|css|js)' parameter in the Blocks constructor",
        category=DeprecationWarning,
    )
    demo = build_ui()

# ==============================================================================
# 9. 启动入口
# ==============================================================================
if __name__ == "__main__":
    print("=" * 64)
    print("  KnowSearch 知识检索问答平台 已启动")
    print(f"  - 大模型     : {OPENAI_MODEL or '(未配置，请在 .env 设置 OPENAI_MODEL)'}")
    print(f"  - API 地址   : {OPENAI_BASE_URL}")
    print(f"  - 嵌入模型   : {EMBED_MODEL}")
    print(f"  - 向量库目录 : {os.path.abspath(CHROMA_DIR)}")
    print("  - 本地访问   : http://127.0.0.1:7860")
    print("  - 公网访问   : 由 cloudflared 隧道提供（单独启动，见项目说明）")
    print("  - 提示       : 若未填写 OPENAI_API_KEY，聊天时会有明确报错提示")
    print("=" * 64)
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),  # 可通过环境变量 GRADIO_SERVER_PORT 覆盖
        inbrowser=True, show_error=True,
        **LAUNCH_UI_ARGS,  # Gradio 6.0+ 时 theme/css/js 在此传入
    )
