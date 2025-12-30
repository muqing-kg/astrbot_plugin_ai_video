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
            accumulated = []
            async for line in resp.content:
                if not line.strip():
                    continue
                line_str = line.decode('utf-8').strip()
                if not line_str.startswith('data:'):
                    continue
                payload_str = line_str.split('data:', 1)[1].strip()
                if payload_str == '[DONE]':
                    break
                try:
                    chunk = json.loads(payload_str)
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if isinstance(content, str):
                            accumulated.append(content)
                except json.JSONDecodeError:
                    continue
            
            full_text = "".join(accumulated)
            return self._extract_video_url(full_text)
        
        def _extract_video_url(self, content: str) -> Optional[str]:
            """从文本中提取视频 URL"""
            # 直接 URL（http 开头）
            if content.strip().startswith("http"):
                return content.strip()
            
            # HTML video 标签
            if "<video" in content and "src=" in content:
                match = re.search(r'<video[^>]*src=["\']([^"\']+)["\']', content, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            # 直接 .mp4 URL
            match = re.search(r'(https?://[^\s<>"\')\\]]+\.mp4[^\s<>"\')\\]*)', content, re.IGNORECASE)
            if match:
                return match.group(1)
            
            # Markdown 链接
            match = re.search(r'!?\[[^\]]*\]\(([^)]+)\)', content, re.IGNORECASE)
            if match:
                return match.group(1)
            
            return None
        
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
            accumulated = []
            async for line in resp.content:
                if not line.strip():
                    continue
                line_str = line.decode('utf-8').strip()
                if not line_str.startswith('data:'):
                    continue
                payload_str = line_str.split('data:', 1)[1].strip()
                if payload_str == '[DONE]':
                    break
                try:
                    chunk = json.loads(payload_str)
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if isinstance(content, str):
                            accumulated.append(content)
                except json.JSONDecodeError:
                    continue
            
            full_text = "".join(accumulated)
            return self._extract_video_url(full_text)
        
        def _extract_video_url(self, content: str) -> Optional[str]:
            """从文本中提取视频 URL"""
            # 直接 URL（http 开头）
            if content.strip().startswith("http"):
                return content.strip()
            
            # HTML video 标签
            if "<video" in content and "src=" in content:
                match = re.search(r'<video[^>]*src=["\']([^"\']+)["\']', content, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            # 直接 .mp4 URL
            match = re.search(r'(https?://[^\s<>"\')\\]]+\.mp4[^\s<>"\')\\]*)', content, re.IGNORECASE)
            if match:
                return match.group(1)
            
            # Markdown 链接
            match = re.search(r'!?\[[^\]]*\]\(([^)]+)\)', content, re.IGNORECASE)
            if match:
                return match.group(1)
            
            return None
        
        async def terminate(self):
            if self.session and not self.session.closed:
                await self.session.close()

    # ==================== 插件初始化 ====================

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.sora_client: Optional[PlatoSoraPlugin.SoraAPIClient] = None
        self.grok_client: Optional[PlatoSoraPlugin.GrokAPIClient] = None
        self._session: Optional[aiohttp.ClientSession] = None  # 通用会话
        self._sora_processing: set = set()  # 防止 Sora 任务重复触发
        self._grok_processing: set = set()  # 防止 Grok 任务重复触发
        self.plugin_data_dir = StarTools.get_data_dir("astrbot_plugin_ai_video")
        self.videos_dir = Path(self.plugin_data_dir) / "videos"
        self.videos_dir.mkdir(exist_ok=True, parents=True)

    async def initialize(self):
        if Image is None:
            logger.warning("Pillow 未安装，无法使用图片比例自动识别功能")
        
        # 创建通用会话
        self._session = aiohttp.ClientSession()
        timeout = self.conf.get("polling_timeout", 300)
        self.polling_interval = self.conf.get("polling_interval", 5)
        
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
        
        text = prompt.strip() if prompt else event.message_str.strip()
        if not text:
            return

        prompt_text, params = self._parse_sora_params(text)
        if not prompt_text:
            return

        can_proceed, error_message = await self._check_permissions(event)
        if not can_proceed:
            if error_message: 
                yield event.plain_result(error_message)
            return
        
        # 并发限制
        user_id = str(event.get_sender_id())
        if user_id in self._sora_processing:
            yield event.plain_result("⚠️ 您已有 Sora 任务在进行中")
            return
        
        self._sora_processing.add(user_id)
        try:
            async for result in self._generate_sora_video(event, prompt_text, params):
                yield result
        finally:
            self._sora_processing.discard(user_id)

        event.stop_event()

    def _parse_sora_params(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """解析 Sora 参数（横/竖屏、时长）"""
        params = {}
        
        if text.startswith("sora"):
            text = text.removeprefix("sora").strip()

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
            else:
                break
        
        prompt = " ".join(parts[prompt_start:]).strip() if prompt_start < len(parts) else ""
        return prompt, params

    async def _generate_sora_video(self, event: AstrMessageEvent, prompt: str, params: Dict[str, Any]):
        """Sora 视频生成核心逻辑"""
        image_bytes = await self._get_image_from_event(event)
        
        duration = params.get('duration', 15)
        duration = min(max(duration, 10), 15)
        
        # 确定模型
        if image_bytes:
            # 图生视频：自动识别图片方向
            orientation = await self._get_aspect_ratio_from_image(image_bytes)
            if not orientation:
                yield event.plain_result("❌ 无法识别图片方向")
                return
            model = f"sora-video-{orientation}-{duration}s"
            logger.info(f"图生视频 - 方向: {orientation}, 时长: {duration}秒, 模型: {model}")
        elif 'orientation' in params:
            # 文生视频：用户指定了方向
            orientation = params['orientation']
            model = f"sora-video-{orientation}-{duration}s"
            logger.info(f"文生视频 - 方向: {orientation}, 时长: {duration}秒, 模型: {model}")
        else:
            # 文生视频：用户未指定方向，使用配置的默认模型
            model = self.conf.get("sora_model", "sora-video-landscape-15s")
            logger.info(f"文生视频 - 使用默认模型: {model}")
        
        yield event.plain_result(f"🎬 正在进行 [{'图生视频' if image_bytes else '文生视频'}] ...")

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
        
        user_id = str(event.get_sender_id())
        if user_id in self._grok_processing:
            yield event.plain_result("⚠️ 您已有 Grok 任务在进行中")
            return
        
        self._grok_processing.add(user_id)
        yield event.plain_result("🎬 正在进行 [图生视频] ...")
        
        try:
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
        
        finally:
            self._grok_processing.discard(user_id)

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
                     "格式：/sora [横/竖] [10/15] 提示词\n"
                     "示例：/sora 横屏 15 一只奔跑的狗\n\n"
                     "图生视频：\n"
                     "格式：/sora [10/15] 提示词 + 图片\n"
                     "• 自动识别图片方向\n\n"
                     "━━━━━━━━━━━━━━\n"
                     "【Grok 使用方法】\n\n"
                     "格式：/grok <提示词> + 图片\n"
                     "示例：/grok 让画面动起来\n")
        yield event.plain_result(help_text)

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
