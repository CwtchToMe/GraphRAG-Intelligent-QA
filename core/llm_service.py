"""
大模型服务模块 - 支持API和本地部署(Ollama)
负责：LLM连接管理、Prompt管理、答案生成

支持两种模式:
1. API模式: DeepSeek/OpenAI API (需要网络+API Key)
2. 本地模式: Ollama + 开源模型 (免费, 无需网络)

自动检测: 优先使用本地Ollama, 失败时回退到API
"""
from typing import Optional, Tuple, Dict
import subprocess
import json


class LLMService:
    """
    大模型服务类 (支持双模式)

    功能:
    - 自动检测Ollama本地模型
    - 支持DeepSeek/OpenAI API
    - Prompt模板管理
    - 答案生成
    """

    def __init__(self,
                 api_key: str = "",
                 api_base: str = "https://api.deepseek.com/v1",
                 model_name: str = "deepseek-chat",
                 local_model: str = "qwen2.5:7b-instruct",
                 temperature: float = 0.7,
                 max_tokens: int = 2000,
                 use_local: bool = None,
                 timeout: int = 60):
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.local_model = local_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.llm = None
        self.is_initialized = False
        self.mode = "unknown"

        # Prompt模板
        self.system_prompt = """你是基于知识图谱的智能问答助手。

【核心原则】
1. **严格基于提供的上下文回答**，不要编造信息。信息不足时说"图谱中没有相关信息"
2. **多跳推理**：当问题包含"还/其他/以及/另外/和谁"等关键词时，必须执行多步推理
3. **两步推理示例**：
   - 问题"周星驰还主演了哪些电影？"
   - 推理：周星驰 --[演员]--> 电影1， 周星驰 --[演员]--> 电影2
   - 回答：周星驰还主演了电影1、电影2等
4. 标注信息来源（知识图谱路径 / 语义相似文档）
5. 回答准确、清晰、有条理

【多跳推理规则】
- 步骤1：找到问题中的核心实体（如人物/电影名）
- 步骤2：沿关系找到关联实体
- 步骤3：再沿关系找到最终答案
- 如果图谱无法支持该推理步骤，明确告知用户

【回答格式示例】
基于知识图谱，我推理得到：
- 周星驰 --[演员]--> 《功夫》
- 周星驰 --[演员]--> 《大话西游》
- 刘镇伟 --[导演]--> 《大话西游》

因此，周星驰还主演了《功夫》等电影，这些信息来自图谱中的三元组。"""

        self.ner_prompt = """你是一个专业的中文命名实体识别专家。

请从给定文本中识别所有实体，并分类为以下类型：
- PERSON: 人物姓名（如：贾宝玉、张三）
- WORK: 作品名称（如：《红楼梦》）
- ORGANIZATION: 组织机构名称（如：荣国府、XX公司）
- LOCATION: 地点名称（如：大观园、北京）
- CONCEPT: 概念/抽象词（如：爱情、家族）
- EVENT: 事件名称（如：元妃省亲）
- DATE: 时间表达（如：2024年1月）

请以JSON数组格式输出：
[{"name": "实体名称", "type": "实体类型"}]

只输出JSON，不要其他文字。"""

        self.re_prompt = """你是一个专业的中文关系抽取专家。

请从文本中识别实体之间的关系，支持的关系类型包括：
- IS_A: 是一种/属于类别
- PART_OF: 属于/组成部分
- LOCATED_IN: 位于
- BORN_IN: 出生于
- RELATIVE_OF: 亲属关系
- LOVER_OF: 爱人/恋人
- FRIEND_OF: 朋友
- MEMBER_OF: 成员
- FOUNDED: 创立
- OCCURRED_AT: 发生于
- RELATED_TO: 相关(通用)

请以JSON数组格式输出：
[{"head": "头实体", "relation": "关系", "tail": "尾实体", "confidence": 0.95}]

只输出JSON，不要其他文字。"""

    def _check_ollama_available(self) -> bool:
        """检查Ollama是否可用 (多种检测方式)"""
        # 方法1: 尝试urllib连接 (无需额外依赖)
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags")
            response = urllib.request.urlopen(req, timeout=5)
            return response.status == 200
        except:
            pass

        # 方法2: 尝试httpx连接
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            pass

        # 方法3: 尝试requests连接
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            pass

        # 方法4: 命令行检测 (备选)
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _check_local_model_exists(self) -> bool:
        """检查指定的本地模型是否存在 (多种检测方式)"""
        # 方法1: 通过urllib HTTP API检测
        try:
            import urllib.request
            import json as _json
            req = urllib.request.Request("http://localhost:11434/api/tags")
            response = urllib.request.urlopen(req, timeout=5)
            if response.status == 200:
                data = _json.loads(response.read().decode())
                models = [m.get('name', '').lower() for m in data.get('models', [])]
                model_base = self.local_model.split(':')[0].lower()
                return any(model_base in m for m in models)
        except:
            pass

        # 方法2: 通过httpx检测
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m.get('name', '').lower() for m in data.get('models', [])]
                model_base = self.local_model.split(':')[0].lower()
                return any(model_base in m for m in models)
        except:
            pass

        # 方法3: 命令行检测
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                models = result.stdout.lower()
                model_base = self.local_model.split(':')[0]
                return model_base in models
        except:
            pass

        return False

    def initialize(self, prefer_local: bool = True) -> bool:
        """
        初始化LLM连接（本地优先，可回退API）

        Args:
            prefer_local: 是否优先使用本地模型

        Returns:
            success: 是否成功
        """
        print("[INFO] 正在初始化LLM服务...")
        print(f"  本地优先: {prefer_local}")
        print(f"  本地模型: {self.local_model}")
        print(f"  API模型: {self.model_name}")

        if prefer_local:
            if self._check_ollama_available() and self._check_local_model_exists():
                if self._init_local_mode():
                    return True
                print("[WARN] 本地模式初始化失败，尝试回退到API模式...")
            else:
                print("[WARN] 本地模式不可用（服务未启动或模型不存在）")

        if self.api_key:
            print("[INFO] 尝试初始化API模式...")
            return self._init_api_mode()

        print("[ERROR] 无法初始化LLM：本地模式不可用且未配置API Key")
        return False

    def _init_local_mode(self) -> bool:
        """初始化本地Ollama模式"""
        try:
            from langchain_ollama import ChatOllama

            print(f"[INFO] 正在连接本地Ollama模型: {self.local_model}")

            self.llm = ChatOllama(
                model=self.local_model,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                request_timeout=self.timeout,
            )

            print(f"[INFO] 测试连接中... (超时: {self.timeout}秒)")
            test_response = self.llm.invoke("你好，请用一句话介绍你自己")

            self.is_initialized = True
            self.mode = "local"
            print(f"[OK] 本地LLM模式启动成功！")
            print(f"     模型: {self.local_model}")
            print(f"     测试回复: {test_response.content[:50]}...")
            return True

        except ImportError:
            print("[WARN] langchain_ollama未安装，正在尝试安装...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                    'langchain-ollama', 'ollama'])
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=self.local_model,
                    temperature=self.temperature,
                    num_predict=self.max_tokens,
                    request_timeout=self.timeout,
                )
                test_response = self.llm.invoke("你好")
                self.is_initialized = True
                self.mode = "local"
                print("[OK] 已安装langchain-ollama并成功连接")
                return True
            except Exception as e:
                print(f"[ERROR] 安装失败: {e}")
                return False

        except Exception as e:
            print(f"[ERROR] 本地LLM初始化失败: {e}")
            print(f"  可能原因: 模型加载超时或内存不足")
            print(f"  建议: 尝试使用更小的模型 (如 qwen2.5:3b-instruct)")
            return False

    def _init_api_mode(self) -> bool:
        """初始化API模式"""
        try:
            from langchain_openai import ChatOpenAI

            print(f"[INFO] 正在连接API模型: {self.model_name}")

            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            test_response = self.llm.invoke("你好")

            self.is_initialized = True
            self.mode = "api"
            print(f"[OK] API LLM模式启动成功！")
            print(f"     模型: {self.model_name}")
            print(f"     API端点: {self.api_base}")
            return True

        except Exception as e:
            print(f"[ERROR] API LLM初始化失败: {e}")
            return False

    def generate_answer(self, question: str, context: str = None, system_prompt: str = None, max_retries: int = 2) -> str:
        """
        生成答案

        Args:
            question: 用户问题
            context: 知识库上下文 (用于RAG问答)
            system_prompt: 自定义系统提示 (用于KG提取等特殊任务)
            max_retries: 最大重试次数

        Returns:
            answer: 生成的答案

        Raises:
            Exception: LLM未初始化或调用超时
        """
        if not self.is_initialized or not self.llm:
            raise Exception("LLM未初始化")

        sys_prompt = system_prompt or self.system_prompt

        if context and not system_prompt:
            user_content = f"""用户问题: {question}

知识库上下文:
{context}

请基于以上上下文回答用户问题，并在回答中标注知识来源。"""
        else:
            user_content = question

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke([
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content}
                ])
                return response.content
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                if 'timeout' in error_msg or 'timed out' in error_msg:
                    print(f"[WARN] LLM调用超时 (尝试 {attempt + 1}/{max_retries + 1})")
                    if attempt < max_retries:
                        print(f"  正在重试...")
                        continue
                    raise Exception(f"LLM调用超时，已重试{max_retries}次。请检查模型是否正常运行。")
                else:
                    raise e

        raise last_error

    def test_connection(self) -> Tuple[bool, str]:
        """
        测试LLM连接

        Returns:
            success: 是否成功
            message: 测试消息或错误信息
        """
        if self._check_local_model_exists():
            if self.initialize(prefer_local=True):
                return True, f"✅ 本地模式连接成功！(模型: {self.local_model})"
            else:
                return False, "❌ 本地模式连接失败"

        if self.api_key:
            if self.initialize(prefer_local=False):
                return True, f"✅ API模式连接成功！(模型: {self.model_name})"
            else:
                return False, "❌ API模式连接失败"

        return False, ("❌ 无法连接:\n"
                      "  - 本地Ollama不可用\n"
                      "  - 未配置API Key\n"
                      "\n请先安装Ollama: https://ollama.com/download")

    def get_prompts(self) -> Dict[str, str]:
        """获取所有Prompt模板"""
        return {
            'system': self.system_prompt,
            'ner': self.ner_prompt,
            're': self.re_prompt
        }

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            'is_initialized': self.is_initialized,
            'mode': self.mode,
            'model': self.local_model if self.mode == 'local' else self.model_name,
            'local_available': self._check_ollama_available(),
            'local_model_exists': self._check_local_model_exists(),
            'api_configured': bool(self.api_key),
        }
