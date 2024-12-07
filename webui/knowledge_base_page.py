import os

import streamlit as st
from utils import PLATFORMS, get_embedding_models, get_kb_names
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import get_embedding_model


def knowledge_base_page():
    if "selected_kb" not in st.session_state:
        st.session_state["selected_kb"] = ""
    st.title("知识库管理")
    kb_names = get_kb_names()
    selected_kb = st.selectbox("请选择知识库",
                               ["新建知识库"] + kb_names,
                               index=kb_names.index(st.session_state["selected_kb"]) + 1
                               if st.session_state["selected_kb"] in kb_names
                               else 0
                               )
    if selected_kb == "新建知识库":
        status_placeholder = st.empty()
        with status_placeholder.status("知识库配置", expanded=True) as s:
            cols = st.columns(2)
            kb_name = cols[0].text_input("请输入知识库名称", placeholder="请使用英文，如：companies_information")
            vs_type = cols[1].selectbox("请选择向量库类型", ["Chroma"])
            st.text_area("请输入知识库描述", placeholder="如：介绍企业基本信息")
            cols = st.columns(2)
            platform = cols[0].selectbox("请选择要使用的 Embedding 模型加载方式", PLATFORMS)
            embedding_models = get_embedding_models(platform)
            embedding_model = cols[1].selectbox("请选择要使用的 Embedding 模型", embedding_models)
            submit = st.button("创建知识库")
            if submit and kb_name.strip():
                kb_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb")
                kb_path = os.path.join(kb_root, kb_name)
                file_storage_path = os.path.join(kb_path, "files")
                vs_path = os.path.join(kb_path, "vectorstore")
                if not os.path.exists(kb_path):
                    os.mkdir(kb_path)
                if not os.path.exists(file_storage_path):
                    os.mkdir(file_storage_path)
                if not os.path.exists(vs_path):
                    os.mkdir(vs_path)
                else:
                    st.error("知识库已存在")
                    s.update(label=f'知识库配置', expanded=True, state="error")
                    st.stop()
                st.success("创建知识库成功")
                s.update(label=f'已创建知识库"{kb_name}"', expanded=False)
                st.session_state["selected_kb"] = kb_name
                st.rerun()
            elif submit and not kb_name.strip():
                st.error("知识库名称不能为空")
                s.update(label=f'知识库配置', expanded=True, state="error")
                st.stop()
    else:
        kb_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb")
        kb_path = os.path.join(kb_root, selected_kb)
        file_storage_path = os.path.join(kb_path, "files")
        vs_path = os.path.join(kb_path, "vectorstore")
        uploader_placeholder = st.empty()

        supported_file_formats = ["md"] #, "txt"
        with uploader_placeholder.status("上传文件至知识库", expanded=True) as s:
            files = st.file_uploader("请上传文件", type=supported_file_formats, accept_multiple_files=True)
            upload = st.button("上传")
        if upload:
            for file in files:
                b = file.getvalue()
                with open(os.path.join(file_storage_path, file.name), "wb") as f:
                    f.write(b)

            from langchain_community.document_loaders import DirectoryLoader, TextLoader
            text_loader_kwargs = {"autodetect_encoding": True}
            loader = DirectoryLoader(
                file_storage_path,
                glob=[f"**/{file.name}" for file in files],
                show_progress=True,
                use_multithreading=True,
                loader_cls=TextLoader,
                loader_kwargs=text_loader_kwargs,
            )
            docs_list = loader.load()

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=500, chunk_overlap=100
            )
            doc_splits = text_splitter.split_documents(docs_list)

            import chromadb.api

            chromadb.api.client.SharedSystemClient.clear_system_cache()

            vectorstore = Chroma(
                collection_name=selected_kb,
                embedding_function=get_embedding_model(model="quentinz/bge-large-zh-v1.5:latest"),
                persist_directory=vs_path,
            )

            vectorstore.add_documents(doc_splits)
            st.success("上传文件成功")

