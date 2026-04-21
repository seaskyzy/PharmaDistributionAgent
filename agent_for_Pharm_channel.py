#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import streamlit as st
from typing import List, Dict, Any
from neo4j import GraphDatabase
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# ==================== 配置区域：从 st.secrets 读取 ====================
try:
    NEO4J_URI = st.secrets["NEO4J_URI"]
    NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
    NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except KeyError as e:
    st.error(f"缺少必要的 secrets 配置: {e}")
    st.stop()

if not NEO4J_PASSWORD or not DEEPSEEK_API_KEY:
    st.error("请在 .streamlit/secrets.toml 中正确配置所有密钥")
    st.stop()

# 设置 DeepSeek API 环境变量（供 langchain 使用）
os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY

# ==================== 1. Neo4j 连接（使用缓存）====================
class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("✅ 成功连接到 Neo4j Aura")

    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

@st.cache_resource
def get_db():
    return Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

db = get_db()

# ==================== 2. 工具定义（完全复制你的实现，修复 filter 中的键名错误）====================
@tool
def find_product_paths(product_code: str) -> str:
    """
    查询某个药品在渠道网络中的所有流通路径。
    返回：JSON 格式的交易列表，每个交易包含发货方、收货方、金额、交易ID。
    """
    query = """
    MATCH (start:Institution_from)-[:Send]->(t:Transaction)-[:To]->(mid:Institution_to)
    WHERE (t)-[:HAS_PRODUCT]->(:Product {product_code: toInteger($product_code)})
    RETURN start.from_ins_name AS from_name, t.trans_id, t.amount, mid.to_ins_name AS to_name
    LIMIT 20
    """
    result = db.run_query(query, {"product_code": product_code})
    if not result:
        return json.dumps([])
    transactions = []
    for record in result:
        transactions.append({
            "from_name": record["from_name"],
            "to_name": record["to_name"],
            "amount": record["t.amount"],
            "trans_id": record["t.trans_id"]
        })
    return json.dumps(transactions, ensure_ascii=False)

@tool
def mark_high_risk_transaction(transaction_id: str) -> str:
    """
    将指定 ID 的交易标记为高风险，并添加审核状态。
    """
    query = """
    MATCH (t:Transaction {trans_id: toInteger($trans_id)})
    SET t.risk_level = 'HIGH', t.review_status = 'PENDING', t.risk_flagged_at = datetime()
    RETURN t.trans_id, t.risk_level, t.review_status
    """
    try:
        result = db.run_query(query, {"trans_id": transaction_id})
        if result:
            return f"成功将交易 {transaction_id} 标记为高风险，审核状态：待审核。"
        else:
            return f"未找到交易 {transaction_id}，标记失败。"
    except Exception as e:
        return f"标记失败：{str(e)}"

@tool
def count_institutions() -> str:
    """统计所有机构（发货方+收货方）的总数。"""
    query_from = "MATCH (i:Institution_from) RETURN count(i) AS cnt"
    query_to = "MATCH (i:Institution_to) RETURN count(i) AS cnt"
    from_count = db.run_query(query_from)[0]["cnt"]
    to_count = db.run_query(query_to)[0]["cnt"]
    total = from_count + to_count
    return f"共有 {from_count} 个发货方机构，{to_count} 个收货方机构，总计 {total} 个。"

@tool
def filter_transactions_by_amount(transactions_json: str, min_amount: float = None, max_amount: float = None) -> str:
    """
    从交易列表中筛选金额在指定范围内的交易。
    参数：
        transactions_json: 由 find_product_paths 返回的 JSON 字符串
        min_amount: 最小金额（可选）
        max_amount: 最大金额（可选）
    """
    try:
        transactions = json.loads(transactions_json)
    except:
        return "错误：无法解析交易数据"
    filtered = []
    for tx in transactions:
        # 注意：这里使用 "amount" 键，与 find_product_paths 返回的字典一致
        amount = tx.get("amount")
        if amount is None:
            continue
        if min_amount is not None and amount < min_amount:
            continue
        if max_amount is not None and amount > max_amount:
            continue
        filtered.append(tx)
    if not filtered:
        return f"未找到金额在 {min_amount or '不限'} ~ {max_amount or '不限'} 之间的交易。"
    return json.dumps(filtered)

# ==================== 3. 创建 Agent（使用缓存）====================
@st.cache_resource
def get_agent():
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=DEEPSEEK_API_KEY,
        temperature=0.9,
        request_timeout=60,
        max_retries=2,
    )
    tools = [find_product_paths, mark_high_risk_transaction, count_institutions, filter_transactions_by_amount]
    system_prompt = """
你是一个药品流通领域的 AI 分析专家。你可以查询 Neo4j 知识图谱中的数据，也可以执行标记高风险交易等操作。
你的任务是根据用户的问题，调用合适的工具来回答或执行操作。

1. 对于任何需要多步推理或计算的问题，**必须先输出“思考：”段落**，列出你的步骤：
   - 我需要哪些数据？
   - 第一步调用什么工具，参数是什么？
   - 从工具返回结果中提取哪些字段？
   - 如何计算或过滤？
   - 最后如何组织答案？

2. 然后调用相应的工具获取数据。

3. 最后基于工具返回的结果，用简洁清晰的中文给出最终答案。

可用工具：
- find_product_paths: 查询药品流通路径。参数：product_code (字符串)
- mark_high_risk_transaction: 标记高风险交易。参数：transaction_id (字符串)
- count_institutions: 统计机构数量。无需参数。
- filter_transactions_by_amount: 从交易列表中筛选金额在指定范围内的交易。参数：transactions_json（JSON字符串）, min_amount（可选）, max_amount（可选）

请严格按照用户指令调用工具，并用简洁的语言返回结果。
"""
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent

# ==================== 4. Streamlit UI ====================
st.set_page_config(page_title="药品流通分析助手", layout="wide")
st.title("💊 药品流通智能分析 Agent")
st.markdown("基于 Neo4j + LangChain + DeepSeek，支持自然语言查询药品流通路径、标记高风险交易、统计机构数量。")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题，例如：产品85的流通路径有哪些？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                agent = get_agent()
                response = agent.invoke({"messages": [("user", prompt)]})
                answer = response["messages"][-1].content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"❌ 出错：{str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# 侧边栏信息
with st.sidebar:
    st.header("🔧 可用工具")
    st.markdown("- `find_product_paths`: 查询药品流通路径")
    st.markdown("- `mark_high_risk_transaction`: 标记高风险交易")
    st.markdown("- `count_institutions`: 统计机构数量")
    st.markdown("- `filter_transactions_by_amount`: 按金额筛选交易")
    st.markdown("---")
    st.caption("注意：请确保 Neo4j 和 DeepSeek API 配置正确。")