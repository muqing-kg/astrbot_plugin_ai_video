import asyncio
import base64
import json
import time
import re
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncGenerator, Tuple
import io


Image = None
try:
    from PIL import Image
except ImportError:
    pass

import aiohttp
import aiofiles
from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core import AstrBotConfig
import astrbot.api.message_components as Comp
from astrbot.core.platform.astr_message_event import AstrMessageEvent


class PlatoSoraPlugin(Star):
    """AI 视频生成插件 - 集成 Sora 和 Grok 双引擎"""
    
    # ==================== 通用媒体处理方法 ====================
    
    async def _download_media(self, url: str) -> Optional[bytes]:
        """通用媒体下载方法"""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        try:
            async with self._session.get(url, timeout=120) as resp:
                resp.raise_for_status()
                return await resp.read()
        except aiohttp.ClientResponseError as e:
            logger.error(f"媒体下载失败: {e.message}")
            return None
        except asyncio.TimeoutError:
            logger.error("媒体下载超时")
            return None
        except Exception as e:
            logger.error(f"媒体下载失败: {e}")
            return None

    async def _load_bytes(self, src: str) -> Optional[bytes]:
        """从 URL/文件/base64 加载字节数据"""
        if Path(src).is_file():
            try:
                async with aiofiles.open(src, 'rb') as f:
                    return await f.read()
            except Exception as e:
                logger.error(f"读取本地文件失败: {src}, error: {e}")
                return None
        elif src.startswith("http"):
            return await self._download_media(src)
        elif src.startswith("base64://"):
            return base64.b64decode(src[9:])
        return None

    async def _find_image_in_segments(self, segments: List[Any]) -> Optional[bytes]:
        """从消息段中查找图片"""
        for seg in segments:
            if isinstance(seg, Comp.Image):
                if seg.url and (img := await self._load_bytes(seg.url)): 
                    return img
                if seg.file and (img := await self._load_bytes(seg.file)): 
                    return img
        return None

    async def _get_image_from_event(self, event: AstrMessageEvent) -> Optional[bytes]:
        """从消息事件中提取图片（支持引用和直接发送）"""
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Reply) and seg.chain:
                if image_bytes := await self._find_image_in_segments(seg.chain):
                    return image_bytes
        return await self._find_image_in_segments(event.message_obj.message)


    async def _get_aspect_ratio_from_image(self, image_bytes: bytes) -> Optional[str]:
        """从图片字节识别方向（横屏/竖屏）"""
        if not Image:
            return None

        def process_image():
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    width, height = img.size
                    if width > 0 and height > 0:
                        return "landscape" if width > height else "portrait"
                    return None
            except Exception as e:
                logger.warning(f"自动识别图片比例失败: {e}")
                return None

        return await asyncio.to_thread(process_image)

    async def _save_and_send_video(self, event: AstrMessageEvent, video_url: str, 
                                    video_bytes: bytes, prefix: str = "video") -> AsyncGenerator:
        """通用视频保存和发送逻辑"""
        video_filename = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
        video_path = self.videos_dir / video_filename
        video_path = video_path.resolve()
        
        try:
            async with aiofiles.open(video_path, 'wb') as f:
                await f.write(video_bytes)
            
            logger.info(f"✅ 视频保存成功: {video_path}")
            
            try:
                video_component = Comp.Video.fromFileSystem(path=str(video_path), name=video_filename)
                yield event.chain_result([video_component])
                logger.info("✅ 视频发送成功")
            except Exception as e:
                logger.error(f"发送视频失败: {e}")
                yield event.plain_result(f"🎬 文件发送失败，请点击链接下载：\n{video_url}")
                
        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            yield event.plain_result(f"❌ 视频处理失败: {str(e)}")
        finally:
            try:
                if video_path.exists():
                    await aiofiles.os.remove(video_path)
                    logger.info(f"已清理临时文件: {video_path}")
            except Exception:
                pass

    # ==================== 流式响应解析方法 ====================
    
    @staticmethod
    def _extract_sse_payload(line_str: str) -> Optional[str]:
        """从 SSE 行中提取数据载荷"""
        line_str = line_str.strip()
        if line_str.startswith('event:') or line_str.startswith(':'):
            return None
        if line_str.startswith('data: '):
            return line_str[6:]
        elif line_str.startswith('data:'):
            return line_str[5:]
        return None
    
    @staticmethod
    def _extract_video_url(content: str) -> Optional[str]:
        """从文本中提取视频 URL"""
        if not content or not content.strip():
            return None
        
        content = content.strip()
        
        # 直接 URL
        if content.startswith(("http://", "https://")):
            return content.split()[0].rstrip('.,;!?)\'"')
        
        # HTML video 标签
        for pattern in [
            r'<video[^>]*src=["\']([^"\']+)["\']',
            r'<source[^>]*src=["\']([^"\']+)["\']',
            r'<video[^>]*>\s*<source[^>]*src=["\']([^"\']+)["\']',
        ]:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1)
        
        # 视频文件扩展名 URL
        match = re.search(r'(https?://[^\s<>"\')\]\\]+\.(?:mp4|webm|mov|avi|mkv)(?:[?][^\s<>"\')\]\\]*)?)', content, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Markdown 链接格式
        match = re.search(r'!?\[[^\]]*\]\(([^)]+)\)', content)
        if match:
            url = match.group(1)
            if url.startswith(("http://", "https://")):
                return url
        
        # 通用 URL 提取
        match = re.search(r'(https?://[^\s<>"\')\]\\]+)', content)
        if match:
            return match.group(1)
        
        return None
    
    @staticmethod
    def _extract_content_from_chunk(chunk: dict) -> Optional[str]:
        """从响应块中提取内容"""
        if chunk.get("choices"):
            choice = chunk["choices"][0]
            if choice.get("delta"):
                return choice["delta"].get("content", "")
            if choice.get("message"):
                return choice["message"].get("content", "")
            if choice.get("text"):
                return choice["text"]
        for key in ("content", "text", "result", "output"):
            if chunk.get(key):
                return chunk[key]
        return None
    
    @staticmethod
    async def _parse_stream_response(resp, client_name: str = "API") -> Tuple[Optional[str], str]:
        """解析响应，自动兼容流式和非流式格式"""
        video_url = None
        accumulated = []
        is_streaming = False
        raw_content = b""
        
        async for line in resp.content:
            raw_content += line
            if not line or not line.strip():
                continue
            
            try:
                line_str = line.decode('utf-8').strip()
            except UnicodeDecodeError:
                continue
            
            payload_str = PlatoSoraPlugin._extract_sse_payload(line_str)
            if payload_str is not None:
                is_streaming = True
                payload_str = payload_str.strip()
                if payload_str in ('[DONE]', 'done', ''):
                    continue
                
                try:
                    chunk = json.loads(payload_str)
                    content = PlatoSoraPlugin._extract_content_from_chunk(chunk)
                    if content:
                        accumulated.append(content)
                        url = PlatoSoraPlugin._extract_video_url(content)
                        if url:
                            video_url = url
                            logger.info(f"[{client_name}] 检测到视频 URL: {url[:100]}...")
                except json.JSONDecodeError:
                    if payload_str.startswith(("http://", "https://")):
                        video_url = payload_str.split()[0]
                        logger.info(f"[{client_name}] 检测到直接 URL: {video_url[:100]}...")
        
        # 非流式响应处理
        if not is_streaming and raw_content:
            try:
                full_json = json.loads(raw_content.decode('utf-8'))
                content = PlatoSoraPlugin._extract_content_from_chunk(full_json)
                if content:
                    accumulated.append(content)
                    video_url = PlatoSoraPlugin._extract_video_url(content)
                    if video_url:
                        logger.info(f"[{client_name}] 非流式响应检测到视频 URL: {video_url[:100]}...")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        
        full_text = "".join(accumulated)
        if not video_url and full_text:
            logger.info(f"[{client_name}] 累积响应: {full_text[:500]}...")
            video_url = PlatoSoraPlugin._extract_video_url(full_text)
        
        return video_url, full_text

    # ==================== Sora API 客户端 ====================
    
    class SoraAPIClient:
        """Sora 视频生成 API 客户端"""
        
        def __init__(self, api_key: str, api_url: str, timeout: int = 300):
            self.api_key = api_key
            self.api_url = api_url
            self.timeout = timeout
            self.max_retries = 3
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.session = aiohttp.ClientSession()
        
        async def generate_video(self, model: str, prompt: str, 
                                  image_bytes: Optional[bytes] = None) -> Tuple[Optional[str], Optional[str]]:
            """调用 Sora API 生成视频，返回 (video_url, error_msg)"""
            messages = []
            if image_bytes:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]})
            else:
                messages.append({"role": "user", "content": prompt})

            logger.info(f"Sora 使用模型: {model}")
            payload = {"model": model, "messages": messages, "stream": True}
            
            for attempt in range(self.max_retries):
                try:
                    async with self.session.post(
                        self.api_url, 
                        headers=self.headers, 
                        json=payload, 
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            return None, f"API 请求失败 (状态码: {resp.status}): {text[:200]}"
                        
                        video_url, result_text = await self._parse_stream_with_wait(resp)
                        if video_url:
                            return video_url, None
                        
                        # 检查是否为 API 错误
                        if result_text and result_text.startswith("API_ERROR:"):
                            return None, result_text[10:]  # 移除前缀，返回错误信息
                        
                        # 日志输出完整响应以便调试
                        if result_text:
                            logger.warning(f"[Sora] 未能从响应中提取视频 URL，完整响应: {result_text[:1000]}")
                        return None, "API 响应中未包含有效视频 URL"
                        
                except asyncio.TimeoutError:
                    if attempt == self.max_retries - 1:
                        return None, f"请求超时 ({self.timeout}秒)"
                    await asyncio.sleep(1)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        return None, f"请求异常: {str(e)}"
                    await asyncio.sleep(1)
            
            return None, "所有重试均失败"
        
        async def _parse_stream_with_wait(self, resp) -> Tuple[Optional[str], str]:
            """解析流式响应，实时检测状态和 URL，返回 (video_url, error_or_text)"""
            video_url = None
            accumulated = []
            chunk_count = 0
            raw_lines = []
            api_error = None
            
            async for line in resp.content:
                if not line or not line.strip():
                    continue
                
                try:
                    line_str = line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    continue
                
                # 记录原始行
                if len(raw_lines) < 20 or any(kw in line_str.lower() for kw in ['url', 'http', 'video', 'mp4', 'error', 'status']):
                    raw_lines.append(line_str[:300])
                
                # 提取 SSE 载荷
                payload_str = PlatoSoraPlugin._extract_sse_payload(line_str)
                if payload_str is None:
                    # 尝试直接解析 JSON
                    if line_str.startswith('{'):
                        try:
                            chunk = json.loads(line_str)
                            # 检查 API 错误
                            if chunk.get("error"):
                                api_error = chunk["error"].get("message", str(chunk["error"]))
                                logger.error(f"[Sora] API 返回错误: {api_error}")
                            content = PlatoSoraPlugin._extract_content_from_chunk(chunk)
                            if content:
                                accumulated.append(content)
                                url = PlatoSoraPlugin._extract_video_url(content)
                                if url:
                                    video_url = url
                                    logger.info(f"[Sora] 从 JSON 提取视频 URL: {url[:100]}...")
                        except json.JSONDecodeError:
                            pass
                    continue
                
                payload_str = payload_str.strip()
                if payload_str in ('[DONE]', 'done', ''):
                    logger.info("[Sora] 流式响应结束")
                    break
                
                chunk_count += 1
                
                try:
                    chunk = json.loads(payload_str)
                    
                    # 检查 API 错误
                    if chunk.get("error"):
                        api_error = chunk["error"].get("message", str(chunk["error"]))
                        logger.error(f"[Sora] API 返回错误: {api_error}")
                        continue
                    
                    content = PlatoSoraPlugin._extract_content_from_chunk(chunk)
                    
                    if content:
                        accumulated.append(content)
                        
                        # 实时检测 URL
                        url = PlatoSoraPlugin._extract_video_url(content)
                        if url:
                            video_url = url
                            logger.info(f"[Sora] 检测到视频 URL: {url[:100]}...")
                        
                        # 输出进度日志
                        if chunk_count % 10 == 0 or any(kw in content.lower() for kw in ['生成', 'generat', 'complet', 'finish', 'url', 'http']):
                            logger.info(f"[Sora] 块 #{chunk_count}: {content[:200]}...")
                            
                except json.JSONDecodeError:
                    if payload_str.startswith(("http://", "https://")):
                        video_url = payload_str.split()[0]
                        logger.info(f"[Sora] 检测到直接 URL: {video_url[:100]}...")
            
            full_text = "".join(accumulated)
            
            # 输出调试信息
            if not video_url:
                logger.warning(f"[Sora] 共收到 {len(raw_lines)} 行原始响应，{chunk_count} 个有效块")
                for i, raw_line in enumerate(raw_lines[:10]):
                    logger.warning(f"[Sora] 原始行 {i+1}: {raw_line}")
                if full_text:
                    logger.warning(f"[Sora] 累积文本: {full_text[:500]}...")
            
            # 如果有 API 错误，优先返回错误信息
            if api_error:
                return None, f"API_ERROR:{api_error}"
            
            return video_url, full_text
        
        async def terminate(self):
            if self.session and not self.session.closed: 
                await self.session.close()

    # ==================== Grok API 客户端 ====================
    
    class GrokAPIClient:
        """Grok 视频生成 API 客户端"""
        
        def __init__(self, api_key: str, api_url: str, model: str, timeout: int = 300):
            self.api_key = api_key
            self.api_url = api_url
            self.model = model
            self.timeout = timeout
            self.max_retries = 3
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.session = aiohttp.ClientSession()
        
        async def generate_video(self, prompt: str, image_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
            """调用 Grok API 生成视频，返回 (video_url, error_msg)"""
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }]
            
            payload = {"model": self.model, "messages": messages, "stream": True}
            
            for attempt in range(self.max_retries):
                try:
                    async with self.session.post(
                        self.api_url, 
                        headers=self.headers, 
                        json=payload, 
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            return None, f"API 请求失败 (状态码: {resp.status}): {text[:200]}"
                        
                        video_url = await self._parse_stream_response(resp)
                        if video_url:
                            return video_url, None
                        return None, "API 响应中未包含有效视频 URL"
                        
                except asyncio.TimeoutError:
                    if attempt == self.max_retries - 1:
                        return None, f"请求超时 ({self.timeout}秒)"
                    await asyncio.sleep(1)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        return None, f"请求异常: {str(e)}"
                    await asyncio.sleep(1)
            
            return None, "所有重试均失败"
        
        async def _parse_stream_response(self, resp) -> Optional[str]:
            """解析流式响应，提取视频 URL"""
            video_url, _ = await PlatoSoraPlugin._parse_stream_response(resp, "Grok")
            return video_url
        
        async def terminate(self):
            if self.session and not self.session.closed:
                await self.session.close()

    # ==================== 插件初始化 ====================

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.sora_client: Optional[PlatoSoraPlugin.SoraAPIClient] = None
        self.grok_client: Optional[PlatoSoraPlugin.GrokAPIClient] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self.plugin_data_dir = StarTools.get_data_dir("astrbot_plugin_ai_video")
        self.videos_dir = Path(self.plugin_data_dir) / "videos"
        self.videos_dir.mkdir(exist_ok=True, parents=True)

    async def initialize(self):
        if Image is None:
            logger.warning("Pillow 未安装，无法使用图片比例自动识别功能")
        
        self._session = aiohttp.ClientSession()
        timeout = 300
        
        # Sora 客户端初始化
        if self.conf.get("sora_enabled", True):
            sora_api_key = self.conf.get("sora_api_key")
            sora_api_url = self.conf.get("sora_api_url")
            if sora_api_key and sora_api_url:
                self.sora_client = self.SoraAPIClient(
                    api_key=sora_api_key, 
                    api_url=sora_api_url, 
                    timeout=timeout
                )
                logger.info("Sora 引擎已加载")
            else:
                logger.warning("Sora 初始化失败: 请检查 sora_api_key 和 sora_api_url 配置")
        
        # Grok 客户端初始化
        if self.conf.get("grok_enabled", True):
            grok_api_key = self.conf.get("grok_api_key")
            grok_api_url = self.conf.get("grok_api_url")
            grok_model = self.conf.get("grok_model", "grok-imagine-0.9")
            if grok_api_key and grok_api_url:
                self.grok_client = self.GrokAPIClient(
                    api_key=grok_api_key, 
                    api_url=grok_api_url, 
                    model=grok_model, 
                    timeout=timeout
                )
                logger.info("Grok 引擎已加载")
            else:
                logger.warning("Grok 初始化失败: 请检查 grok_api_key 和 grok_api_url 配置")
        
        logger.info("AI 视频生成插件初始化完成")

    # ==================== Sora 命令 ====================

    @filter.command("sora")
    async def on_sora_request(self, event: AstrMessageEvent, *, prompt: str = ""):
        """Sora 视频生成：/sora [横/竖] [10/15] <提示词>"""
        if not self.conf.get("sora_enabled", True):
            yield event.plain_result("❌ Sora 视频生成功能已关闭")
            return
        
        if not self.sora_client:
            yield event.plain_result("❌ Sora 客户端未初始化，请检查配置")
            return
        # 优先使用 event.message_str 获取完整消息
        text = event.message_str.strip()
        if not text:
            return

        logger.info(f"[Sora] 原始输入: '{text}'")
        prompt_text, params = self._parse_sora_params(text)
        logger.info(f"[Sora] 解析结果: prompt='{prompt_text}', params={params}")
        if not prompt_text:
            logger.warning(f"[Sora] 解析后 prompt 为空，忽略命令")
            return

        can_proceed, error_message = await self._check_permissions(event)
        if not can_proceed:
            if error_message: 
                yield event.plain_result(error_message)
            return
        
        async for result in self._generate_sora_video(event, prompt_text, params):
            yield result

        event.stop_event()

    # 视频风格信息（风格ID -> (中文名, 说明)）
    STYLE_INFO = {
        "festive": ("节日", "🎉 节日庆典风格，充满欢乐气氛"),
        "kakalaka": ("混沌", "🪭 混沌艺术风格，独特视觉效果"),
        "news": ("新闻", "📺 新闻播报风格，正式专业"),
        "selfie": ("自拍", "🤳 自拍视角风格，第一人称"),
        "handheld": ("手持", "📱 手持拍摄风格，真实抖动感"),
        "golden": ("金色", "✨ 金色调风格，华丽高贵"),
        "anime": ("动漫", "🎌 日式动漫风格，二次元画风"),
        "retro": ("复古", "📼 复古怀旧风格，老电影质感"),
        "nostalgic": ("怀旧", "🎞️ 老照片风格，泛黄胶片感"),
        "comic": ("漫画", "💥 漫画风格，分镜画格效果"),
    }

    # 风格别名映射（中文/英文名 -> 风格ID）
    STYLE_MAP = {
        # 中文别名
        "节日": "festive", "节庆": "festive",
        "混沌": "kakalaka",
        "新闻": "news",
        "自拍": "selfie",
        "手持": "handheld",
        "金色": "golden",
        "动漫": "anime", "动画": "anime",
        "复古": "retro",
        "怀旧": "nostalgic", "老照片": "nostalgic",
        "漫画": "comic",
        # 英文名称
        "festive": "festive",
        "kakalaka": "kakalaka",
        "news": "news",
        "selfie": "selfie",
        "handheld": "handheld",
        "golden": "golden",
        "anime": "anime",
        "retro": "retro",
        "nostalgic": "nostalgic", "vintage": "nostalgic",
        "comic": "comic",
    }

    def _parse_sora_params(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """解析 Sora 参数（横/竖屏、时长、风格）"""
        params = {}
        
        # 移除命令前缀
        for prefix in ["/sora", "sora"]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break

        parts = text.split()
        prompt_start = 0
        
        for i, part in enumerate(parts):
            p = part.lower()
            if p in ["横", "横屏", "landscape"]:
                params['orientation'] = "landscape"
                prompt_start = i + 1
            elif p in ["竖", "竖屏", "portrait"]:
                params['orientation'] = "portrait"
                prompt_start = i + 1
            elif p in ["10", "10s"]:
                params['duration'] = 10
                prompt_start = i + 1
            elif p in ["15", "15s"]:
                params['duration'] = 15
                prompt_start = i + 1
            elif p in ["25", "25s"]:
                params['duration'] = 25
                prompt_start = i + 1
            elif p in self.STYLE_MAP:
                params['style'] = self.STYLE_MAP[p]
                prompt_start = i + 1
            else:
                break
        
        prompt = " ".join(parts[prompt_start:]).strip() if prompt_start < len(parts) else ""
        return prompt, params

    async def _generate_sora_video(self, event: AstrMessageEvent, prompt: str, params: Dict[str, Any]):
        """Sora 视频生成核心逻辑"""
        image_bytes = await self._get_image_from_event(event)
        
        duration = params.get('duration', 15)
        duration = min(max(duration, 10), 25)
        style = params.get('style')
        
        # ===== 确定生成模式 =====
        
        user_orientation = params.get('orientation')
        
        # 图生视频（引用或发送了图片）
        if image_bytes:
            auto_orientation = await self._get_aspect_ratio_from_image(image_bytes)
            orientation = user_orientation or auto_orientation or 'landscape'
            model = f"sora2-{orientation}-{duration}s"
            mode_name = "图生视频"
        
        # 文生视频（纯文本）
        else:
            # 如果用户指定了参数，动态生成模型名；否则使用配置的默认模型
            if user_orientation or 'duration' in params:
                orientation = user_orientation or 'landscape'
                model = f"sora2-{orientation}-{duration}s"
            else:
                model = self.conf.get("sora_default_model", "sora2-landscape-15s")
            mode_name = "文生视频"
        
        # 应用风格（在 prompt 前添加 {风格ID}）
        style_name = None
        if style:
            prompt = f"{{{style}}}{prompt}"
            style_name = self.STYLE_INFO.get(style, (style,))[0]  # 获取中文名
            logger.info(f"[{mode_name}] 风格: {style_name}")
        
        logger.info(f"[{mode_name}] 方向: {orientation}, 时长: {duration}秒, 模型: {model}")
        yield event.plain_result(f"🎬 正在进行 [{mode_name}]{' (' + style_name + '风格)' if style_name else ''} ...")

        # 调用 API（统一的同步接口）
        video_url, error_msg = await self.sora_client.generate_video(
            model=model, prompt=prompt, image_bytes=image_bytes
        )
        
        if error_msg:
            yield event.plain_result(f"❌ 生成失败: {error_msg}")
            return

        if not video_url:
            yield event.plain_result("❌ 未能获取到视频 URL")
            return
        
        # 下载并发送
        logger.info(f"正在下载视频: {video_url}")
        video_bytes = await self._download_media(video_url)
        
        if video_bytes:
            async for result in self._save_and_send_video(event, video_url, video_bytes, "sora"):
                yield result
        else:
            yield event.plain_result(f"❌ 视频下载失败，链接: {video_url}")

    # ==================== Grok 命令 ====================

    @filter.command("grok")
    async def on_grok_request(self, event: AstrMessageEvent, *, prompt: str = ""):
        """Grok 图生视频：/grok <提示词>（需带图片）"""
        if not self.conf.get("grok_enabled", True):
            yield event.plain_result("❌ Grok 视频生成功能已关闭")
            return
        
        if not self.grok_client:
            yield event.plain_result("❌ Grok 客户端未初始化，请检查配置")
            return
        
        # 检查提示词
        prompt = prompt.strip()
        if not prompt:
            yield event.plain_result("❌ 请输入提示词，例如：/grok 让画面动起来")
            return
        
        can_proceed, error_message = await self._check_permissions(event)
        if not can_proceed:
            if error_message:
                yield event.plain_result(error_message)
            return
        
        image_bytes = await self._get_image_from_event(event)
        if not image_bytes:
            yield event.plain_result("❌ Grok 需要图片，请上传或引用图片")
            return
        
        yield event.plain_result("🎬 正在进行 [图生视频] ...")
        
        video_url, error_msg = await self.grok_client.generate_video(prompt, image_bytes)
        
        if error_msg:
            yield event.plain_result(f"❌ 生成失败: {error_msg}")
            return
        
        if not video_url:
            yield event.plain_result("❌ 未能获取到视频 URL")
            return
        
        logger.info(f"正在下载视频: {video_url}")
        video_bytes = await self._download_media(video_url)
        
        if video_bytes:
            async for result in self._save_and_send_video(event, video_url, video_bytes, "grok"):
                yield result
        else:
            yield event.plain_result(f"❌ 视频下载失败，链接: {video_url}")

    # ==================== 帮助命令 ====================

    @filter.command("视频帮助", prefix_optional=True)
    async def on_cmd_help(self, event: AstrMessageEvent):
        help_text = ("【AI 视频生成使用说明】\n\n"
                     "🎬 支持的引擎：\n"
                     "• Sora - 文生视频 + 图生视频\n"
                     "• Grok - 仅图生视频\n\n"
                     "━━━━━━━━━━━━━━\n"
                     "【Sora 使用方法】\n\n"
                     "文生视频：\n"
                     "格式：/sora [横/竖] [10/15/25] [风格] 提示词\n"
                     "示例：/sora 横 15 动漫 一只奔跑的狗\n\n"
                     "图生视频：\n"
                     "格式：/sora [10/15/25] [风格] 提示词 + 图片\n"
                     "• 自动识别图片方向\n\n"
                     "💡 发送 /sora风格 查看所有预设风格\n"
                     "📌 不指定风格则使用默认效果\n\n"
                     "━━━━━━━━━━━━━━\n"
                     "【Grok 使用方法】\n\n"
                     "格式：/grok <提示词> + 图片\n"
                     "示例：/grok 让画面动起来\n")
        yield event.plain_result(help_text)

    @filter.command("sora风格")
    async def on_style_list(self, event: AstrMessageEvent):
        """查看 Sora 视频风格列表"""
        lines = ["【Sora 视频风格列表】\n"]
        for style_id, (name, desc) in self.STYLE_INFO.items():
            lines.append(f"• {name} ({style_id})\n  {desc}")
        lines.append("\n使用方法：/sora [风格] 提示词")
        lines.append("示例：/sora 动漫 一只猫在奔跑")
        yield event.plain_result("\n".join(lines))

    # ==================== 权限检查 ====================

    async def _check_permissions(self, event: AstrMessageEvent) -> Tuple[bool, Optional[str]]:
        """检查用户和群组权限"""
        # 群组黑名单
        group_blacklist = self.conf.get("group_blacklist", [])
        if hasattr(event, 'get_group_id') and group_blacklist:
            try:
                group_id = event.get_group_id()
                if group_id and group_id in group_blacklist:
                    return False, None
            except:
                pass
        
        # 群组白名单
        group_whitelist = self.conf.get("group_whitelist", [])
        if hasattr(event, 'get_group_id') and group_whitelist:
            try:
                group_id = event.get_group_id()
                if group_id and group_id not in group_whitelist:
                    return False, None
            except:
                pass
        
        # 用户黑名单
        user_blacklist = self.conf.get("user_blacklist", [])
        if event.get_sender_id() in user_blacklist:
            return False, None
            
        # 用户白名单
        user_whitelist = self.conf.get("user_whitelist", [])
        if user_whitelist and event.get_sender_id() not in user_whitelist:
            return False, None
            
        return True, None

    # ==================== 插件清理 ====================

    async def terminate(self):
        if self.sora_client: 
            await self.sora_client.terminate()
        if self.grok_client:
            await self.grok_client.terminate()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("AI 视频生成插件已终止")
