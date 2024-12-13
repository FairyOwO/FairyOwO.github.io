> 本文的框架是 `llama_index`

> 整理参考自
> [EasyRAG](https://github.com/BUAADreamer/EasyRAG)
> [RAG 最佳实践](https://zhuanlan.zhihu.com/p/8861103446)

## 数据读取与处理

将其他类型转换为几种基础类型, 再将对基础类型进行解析

使用到的基础类型为: `markdown`, `html`, `pdf`

使用到的库为: [dify_rag](https://github.com/hustyichi/dify-rag)
是一个不错的基础类型解析方案, 解析成 `langchain` 的 `Document` 形式
> 尽管他是给 `dify` 设计, 但他是通用式设计, 可以用在其他地方

```sh
pip install dify_rag
```

```python
from dify_rag.extractor.html_extractor import HtmlExtractor
from dify_rag.extractor.markdown_extractor import MarkdownExtractor

documents = HtmlExtractor(r'path/to/data.html').extract()  # MarkdownExtractor(r'path/to/data.md').extract()

# 转换成 llama_index 的 Document 格式
docs = []
for i in documents:
    i.metadata['titles'] = str(i.metadata['titles'])  # 兼容性问题, llama_index 不支持 titles 里用list[str]
    docs.append(Document(text=i.page_content, metadata=i.metadata))

```

## 切分文档

`chunk_size` 在 256 512 1024 中选择, 这里选择的是 512
`chunk_overlap` 视存储成本, 这里选择是 40

### 父子切分

> 将一份 Document 中的文档, 切分成父子的形式, 检索到子节点的时候, 使用父节点返回, 来拓展上下文

```python
node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[512 * 4, 512], chunk_overlap=40)
nodes = node_parser.get_nodes_from_documents(docs)
```

### 普通切分

```python
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=40)
nodes = node_parser.get_nodes_from_documents(docs)
```

### embedding

最好的仍然是使用 `llm` 调整成的 `embedding`, 不过这里考虑到推理成本, 这里使用的是[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

```python
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", cache_folder="cache")
```

TODO