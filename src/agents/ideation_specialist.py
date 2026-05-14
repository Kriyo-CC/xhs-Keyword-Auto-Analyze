"""IdeationSpecialist — LLM 1：爆款选题与内容策划专家。

基于用户痛点、需求和高频提问，反向推导下一篇可能爆火的帖子选题。
输入：InsightRecord + 标准化后的帖子数据
输出：IdeationResult（3-5 个 TopicIdea）
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.llm.client import BaseLLMClient, extract_json_from_text
from src.schemas import InsightRecord, PostRecord
from src.schemas.new_content_agents import IdeationResult, TopicIdea

logger = logging.getLogger(__name__)

_IDEATION_PROMPT = """你是一位深谙小红书爆款逻辑的资深内容总监。你的任务是基于我提供的用户洞察和历史爆款数据，策划3-5个具有极高爆款潜力的新选题。

【输入数据】
用户洞察：
- 用户痛点: {pain_points}
- 用户需求: {user_needs}
- 市场信号: {market_signals}

历史爆款原帖：
{posts_data}

【策划要求】
1. 痛点反转：从用户的痛点（如"吸尘器太大"、"吸台面手累"、"费瓷砖"）切入，提供明确的解决方案。
2. 场景细分：切入具体生活场景（如：如何清理竖放的书籍、有鼻炎的家庭如何大扫除）。
3. 情绪价值：标题和立意要带有小红书特有的"相见恨晚"、"新大陆"、"懒人福音"等情绪色彩。

【输出格式】
请以 JSON 格式输出，包含以下字段：
{{
  "topic_ideas": [
    {{
      "topic_title": "爆款标题（需包含网感词汇和痛点，控制在20字以内）",
      "target_audience": "目标人群标签（如：鼻炎患者、独居懒人等）",
      "core_hook": "核心抓手（用一句话说明为什么这个选题会火，命中了哪个强需求）",
      "content_framework": "内容大纲（包含引子、痛点共鸣、解决方案、互动提问）"
    }}
  ]
}}

输出严格 JSON："""


def _build_posts_text(posts: list[PostRecord], max_posts: int = 10) -> str:
    """将帖子数据格式化为 prompt 可用的文本。"""
    if not posts:
        return "无历史帖子数据"

    lines: list[str] = []
    for i, p in enumerate(posts[:max_posts], 1):
        title = (p.title or "").strip()
        content = (p.content or "").strip()
        tags_str = "、".join(p.tags[:5]) if p.tags else "无"
        lines.append(
            f"帖子{i}：\n"
            f"  标题：{title}\n"
            f"  内容：{content[:200]}\n"
            f"  标签：{tags_str}\n"
            f"  互动：{p.likes}赞 {p.comments}评 {p.favorites}藏"
        )

    return "\n".join(lines)


class IdeationSpecialist:
    """LLM 1：爆款选题与内容策划专家。

    参数：
        llm_client: LLM 客户端实例
        keyword: 分析关键词（可选，用于日志）
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        keyword: str = "",
    ):
        self._llm = llm_client
        self._keyword = keyword

    def execute(
        self,
        insight: InsightRecord,
        posts: list[PostRecord],
        keyword: str = "",
    ) -> IdeationResult:
        """执行爆款选题策划。

        参数：
            insight: 洞察记录
            posts: 标准化后的帖子列表
            keyword: 关键词

        返回：
            IdeationResult
        """
        kw = keyword or self._keyword
        logger.info("IdeationSpecialist: keyword=%s, posts=%d", kw, len(posts))

        if not self._llm:
            logger.warning("IdeationSpecialist: llm_client 为空，返回空结果")
            return IdeationResult()

        pain_points = "、".join(insight.pain_points[:10]) if insight.pain_points else "无"
        user_needs = "、".join(insight.user_needs[:10]) if insight.user_needs else "无"
        market_signals = "、".join(insight.market_signals[:10]) if insight.market_signals else "无"
        posts_text = _build_posts_text(posts)

        prompt = _IDEATION_PROMPT.format(
            pain_points=pain_points,
            user_needs=user_needs,
            market_signals=market_signals,
            posts_data=posts_text,
        )

        try:
            text = self._llm.generate(prompt)
            raw = extract_json_from_text(text)
            if raw is None:
                raise RuntimeError(f"IdeationSpecialist: LLM 返回非法 JSON，原始内容: {text[:500]}")

            topic_ideas_raw = raw.get("topic_ideas", [])
            if not isinstance(topic_ideas_raw, list):
                raise ValueError(f"IdeationSpecialist: topic_ideas 非 list: {type(topic_ideas_raw)}")

            topic_ideas = [
                TopicIdea(
                    topic_title=t.get("topic_title", ""),
                    target_audience=t.get("target_audience", ""),
                    core_hook=t.get("core_hook", ""),
                    content_framework=t.get("content_framework", ""),
                )
                for t in topic_ideas_raw
                if isinstance(t, dict)
            ]

            logger.info("IdeationSpecialist: 生成 %d 个选题", len(topic_ideas))
            return IdeationResult(topic_ideas=topic_ideas)

        except Exception as e:
            logger.error("IdeationSpecialist generate 失败: %s", e)
            return IdeationResult()
