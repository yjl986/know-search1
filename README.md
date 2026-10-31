# KnowSearch 知识检索问答平台

基于 **Python + Gradio + ChromaDB + bge 中文嵌入模型 + OpenAI 兼容大模型** 的单文件 RAG 知识问答系统。全部后端逻辑、Web 界面与本地向量库集中于一个 `app.py`，安装依赖后一条命令即可启动，无需部署数据库或中间件服务。

## 一、项目介绍

### 1.1 项目定位
面向个人与小型团队，在本地 / 私有环境对自有文档（PDF、TXT、Markdown、DOCX）进行自然语言问答的平台。文档数据仅存于本机，答案可溯源，支持多用户数据隔离与多模型一键切换。

### 1.2 核心功能
- **文档上传与解析**：支持 PDF / TXT / Markdown / DOCX，可拖拽或点击上传；文本自动探测 UTF-8 / GBK 编码；扫描图片型 PDF 明确报错
- **智能切片与向量化**：段落优先合并 + 滑动窗口兜底（`chunk_size=500`、`overlap=50`），bge 中文模型向量化后写入本地 ChromaDB
- **自然语言问答**：Top-5 向量召回（余弦相似度）+ 大模型流式生成，回答自动标注 `[编号]` 出处并附引用来源卡片（文档名 / 片段号 / 相似度 / 摘录）
- **多轮对话**：保持最近 6 轮上下文连贯；会话持久化（最多保留 50 个）；支持历史问题检索
- **知识库管理**：文档列表、单文档删除、清空知识库（二次确认）、文档内容预览
- **多用户账号**：注册 / 登录 / 退出，PBKDF2-SHA256（120000 次迭代）密码哈希，演示账号 `demo/demo123456`，用户数据按 `user` 元数据隔离
- **AI 模型配置**：界面内修改 API Key / 接口 / 模型，立即生效并持久化；内置 8 个预置档案一键切换；连接测试
- **公网与移动端**：支持 Cloudflare Tunnel 公网访问；响应式布局适配手机整页滚动

## 二、技术架构

### 2.1 总体架构
系统自上而下分为五层：**公网访问层**（可选，Cloudflare Tunnel）→ **展示层**（Gradio 5.x）→ **应用逻辑层**（解析 / 切片 / 检索 / 生成）→ **数据存储层**（ChromaDB 向量库 + JSON 数据文件）→ **模型服务层**（bge 嵌入模型 + OpenAI 兼容大模型）。

### 2.2 数据流
- **入库链路**：上传文档 → `parse_document` 解析为纯文本 → `chunk_text` 智能切片 → `BGEEmbeddingFunction` 向量化 → 写入 ChromaDB（携带 user / doc_id / chunk_index 等元数据）
- **问答链路**：用户提问 → `retrieve` 在当前用户知识库内余弦召回 Top-5 → 组装 Prompt（系统约束 + 最近 6 轮历史 + 编号参考资料）→ 流式调用大模型 → 实时渲染回答并附引用卡片

### 2.3 技术选型
| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Gradio | 5.x（实测 5.50.0） | Python 原生 Web UI，单文件即写即用 |
| ChromaDB | 0.5.23 | 本地持久化，`hnsw:space=cosine` 余弦检索 |
| BAAI/bge-small-zh-v1.5 | sentence-transformers 3.x | 中文嵌入，512 维、约 90MB，`normalize_embeddings=True`，本地 models/ 可离线 |
| openai | 1.40–2.0 | OpenAI 兼容 chat.completions 流式接入 |
| pypdf / python-docx | 4–6 / 1.1–2.0 | PDF 与 DOCX 文本提取 |
| Python | 3.10–3.12（实测 3.11.17） | conda 环境 knowsearch |

### 2.4 核心参数
`CHUNK_SIZE=500`、`CHUNK_OVERLAP=50`、`TOP_K=5`（查询候选 10 条）、`PER_DOC_LIMIT=3`（定向提问放宽）、`temperature=0.3`、历史 6 轮、`SUPPORTED_EXTS=.pdf/.txt/.md/.markdown/.docx`

### 2.5 检索与生成优化
- 检索三重优化：多召回裁剪（10 → 5）、片段级去重、单文档限流
- 相似度分级：高 ≥0.85 / 中 ≥0.70 / 低，帮助判断依据可靠度
- 生成约束：系统提示词强制"仅依据参考资料回答、不得编造、按编号标注出处"；深度思考模型（DeepSeek-Reasoner）`reasoning_content` 流式可视化

## 三、安装运行指南

### 3.1 环境要求
- Python 3.10 – 3.12（建议独立 conda/venv 环境）
- 首次运行需联网下载嵌入模型（约 90MB），之后可完全离线

### 3.2 安装依赖
```bash
pip install -r requirements.txt
```
> 网络较慢时可先手动安装 CPU 版 PyTorch：
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`
> 注意：依赖版本已锁定为一套互相兼容的组合（chromadb 0.5.23 与 transformers 4.46.3 / tokenizers 0.20.3），请勿单独升级其中的包。

### 3.3 配置大模型
```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux / macOS
```
编辑 `.env`，填写 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`（也可启动后在页面"AI 模型配置"中修改，立即生效并持久化）。

### 3.4 启动
```bash
python app.py
```
浏览器自动打开 http://127.0.0.1:7860。

### 3.5 公网访问（可选）
使用 Cloudflare Tunnel 将本地服务暴露到公网：
- 在 Cloudflare 控制台创建隧道（如 know-search），将域名记录类型设为 **Tunnel** 指向该隧道
- 本机运行 cloudflared：`cloudflared tunnel --protocol http2 run --token-file <token文件>`
- 配置 Windows 服务与计划任务实现开机自启与后端自动拉起

### 3.6 常见问题
- **PDF 无法入库**：需为文本型 PDF，扫描图片型会明确报错
- **首次启动慢**：正在下载嵌入模型，等待完成即可
- **清空数据**：向量库在 `chroma_db/`，删除该目录即清空全部知识数据（页面内也有"清空知识库"按钮）
- **同名文档重复上传**：自动覆盖旧版本，无重复切片

## 四、团队信息

| 项目 | 内容 |
| --- | --- |
| 团队 / 个人名称 | 李佳原 |
| 团队成员 | 燕志林 江明哲 |
| 联系方式 | 15738459570 |
| 项目地址 | 【待补充，如 GitHub 仓库链接】 |
