"""CopywritingExpert — LLM 2：小红书"网感"文案写手。

根据 LLM 1 生成的一个选题，结合评论区用户原话（"神评论"），
直接生成可发布的小红书图文/视频脚本。
输入：一个 TopicIdea（来自 LLM 1） + 标准化后的评论数据
输出：CopywritingResult（结构化 JSON）
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.llm.client import BaseLLMClient, extract_json_from_text
from src.schemas import CommentRecord
from src.schemas.new_content_agents import (
    CopywritingResult,
    CopySection,
    PainPointSolution,
    TopicIdea,
)

logger = logging.getLogger(__name__)

_COPYWRITING_PROMPT = """你是一位拥有百万粉丝的小红书金牌文案，擅长运用emoji、分段排版以及真实接地气的口吻进行"种草"。

【输入数据】
选定主题：
- 标题：{topic_title}
- 目标人群：{target_audience}
- 核心抓手：{core_hook}
- 内容大纲：{content_framework}

真实用户评论库：
{comments_data}

【创作要求】
1. 标题：采用双标题或悬念式标题，必须包含适当的 emoji。
2. 引入：使用评论库中用户的真实表述（如"打开新世界"、"越扫越脏"）作为开头，拉近距离。
3. 痛点与干货：列出用户的常见踩坑点，并给出具体解决方案（如推荐手持款、静电拖把组合等）。
4. 排版：句子简短，多用空行，要点必须使用 emoji 作为列表符号。
5. 互动提示：在文末设置钩子，引导用户在评论区提问或分享（例如："你家打扫有什么神器？来抄作业！"）。

【输出格式】
请严格以 JSON 格式输出，不要输出 Markdown，不要添加代码块标记以外的内容：

{{
  "title": "爆款标题（含 emoji，双标题或悬念式，20 字左右）",
  "opening_hook": "引入段落，引用真实用户评论打开话题",
  "pain_point_solutions": [
    {{"pain_point": "用户痛点描述", "solution": "对应解决方案"}}
  ],
  "dry_goods": [
    "干货要点1（含 emoji 列表符号）",
    "干货要点2"
  ],
  "body_sections": [
    {{"heading": "小节标题（可选，用于较长的正文分段）", "content": "小节正文"}}
  ],
  "call_to_action": "文末互动钩子，引导评论和分享",
  "hashtags": ["标签1", "标签2", "标签3"]
}}

输出严格 JSON："""


def _build_comments_text(comments: list[CommentRecord], max_comments: int = 20) -> str:
    """将评论数据格式化为 prompt 可用的文本。"""
    if not comments:
        return "无评论数据"

    sorted_comments = sorted(comments, key=lambda c: c.likes, reverse=True)
    lines: list[str] = []
    for c in sorted_comments[:max_comments]:
        content = (c.content or "").strip()
        if not content:
            continue
        likes_str = f"（{c.likes}赞）" if c.likes > 0 else ""
        lines.append(f"- {content}{likes_str}")

    return "\n".join(lines) if lines else "无有效评论"


class CopywritingExpert:
    """LLM 2：小红书"网感"文案写手。

    参数：
        llm_client: LLM 客户端实例
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm = llm_client

    def execute(
        self,
        selected_topic: TopicIdea,
        comments: list[CommentRecord],
    ) -> CopywritingResult:
        """执行文案生成。

        参数：
            selected_topic: LLM 1 输出的单个选题
            comments: 标准化后的评论列表

        返回：
            CopywritingResult（结构化字段）
        """
        logger.info(
            "CopywritingExpert: topic=%s, comments=%d",
            selected_topic.topic_title, len(comments),
        )

        if not self._llm:
            logger.warning("CopywritingExpert: llm_client 为空，返回空结果")
            return CopywritingResult()

        comments_text = _build_comments_text(comments)

        prompt = _COPYWRITING_PROMPT.format(
            topic_title=selected_topic.topic_title,
            target_audience=selected_topic.target_audience,
            core_hook=selected_topic.core_hook,
            content_framework=selected_topic.content_framework,
            comments_data=comments_text,
        )

        try:
            text = self._llm.generate(prompt)
            raw = extract_json_from_text(text)
            if raw is None:
                logger.warning(
                    "CopywritingExpert: LLM 返回非法 JSON，退化到原始文本。"
                    " 原始内容前 300 字符: %s", text[:300],
                )
                return CopywritingResult(
                    selected_topic_title=selected_topic.topic_title,
                )

            # 解析结构化字段
            pain_point_solutions = [
                PainPointSolution(
                    pain_point=item.get("pain_point", ""),
                    solution=item.get("solution", ""),
                )
                for item in raw.get("pain_point_solutions", [])
                if isinstance(item, dict)
            ]
            dry_goods = [
                item for item in raw.get("dry_goods", [])
                if isinstance(item, str) and item.strip()
            ]
            body_sections = [
                CopySection(
                    heading=item.get("heading", ""),
                    content=item.get("content", ""),
                )
                for item in raw.get("body_sections", [])
                if isinstance(item, dict)
            ]
            hashtags = [
                t.lstrip("#") for t in raw.get("hashtags", [])
                if isinstance(t, str) and t.strip()
            ]

            result = CopywritingResult(
                title=raw.get("title", ""),
                opening_hook=raw.get("opening_hook", ""),
                pain_point_solutions=pain_point_solutions,
                dry_goods=dry_goods,
                body_sections=body_sections,
                call_to_action=raw.get("call_to_action", ""),
                hashtags=hashtags,
                selected_topic_title=selected_topic.topic_title,
            )

            logger.info(
                "CopywritingExpert: 解析成功 title=%s, pain_points=%d, "
                "dry_goods=%d, hashtags=%d",
                result.title[:30] if result.title else "(空)",
                len(result.pain_point_solutions),
                len(result.dry_goods),
                len(result.hashtags),
            )
            return result

        except Exception as e:
            logger.error("CopywritingExpert generate 失败: %s", e)
            return CopywritingResult()
