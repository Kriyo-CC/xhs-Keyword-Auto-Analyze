"""MarketIntelligenceAnalyst — LLM 3：产品口碑与竞品情报分析师。

提取商业价值。分析用户提到的替代方案、未被满足的需求
以及对特定品牌（如戴森）的看法，输出产品改进或选品建议。
输入：InsightRecord + 标准化后的评论数据
输出：MarketIntelligenceResult（结构化 JSON）
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.llm.client import BaseLLMClient, extract_json_from_text
from src.schemas import CommentRecord, InsightRecord
from src.schemas.new_content_agents import (
    BizRecommendation,
    CompetitorItem,
    MarketIntelligenceResult,
    PainPointItem,
)

logger = logging.getLogger(__name__)

_MARKET_INTEL_PROMPT = """你是一位顶级的消费品市场研究员，擅长从社交媒体的碎片化评论中挖掘商业机会、选品方向和产品迭代建议。

【输入数据】
评论区洞察摘要：
- 用户痛点: {pain_points}
- 用户投诉: {complaints}
- 方案提及: {solutions}
- 市场信号: {market_signals}

用户评论明细：
{comments_data}

【分析要求】
请对上述数据进行深度剖析，并提供一份针对"居家清洁品类"的商业情报报告。报告需包含：
1. 现有产品痛点分析：用户对目前吸尘器的核心不满是什么？（如：体积、对材质的损伤、卫生交叉感染顾虑）。
2. 热门平替/竞品词云：用户在评论区自发推荐了哪些其他解决方案或品牌？（如：花王静电除尘掸、擦窗机器人等）。
3. 商业变现/选品建议：基于求链接、求型号的高频信号，如果我们要带货或研发新品，最推荐的3个细分品类是什么？为什么？

【输出格式】
请严格以 JSON 格式输出，不要输出 Markdown，不要添加代码块标记以外的内容。要求分析客观、敏锐且具有可落地性。

{{
  "executive_summary": "2-4 句话的总体摘要，概括核心发现",
  "pain_point_analysis": [
    {{
      "issue": "具体产品痛点",
      "severity": "高/中/低",
      "user_voice": "引用真实用户评论作为证据"
    }}
  ],
  "competitor_landscape": [
    {{
      "name": "替代品或品牌名称",
      "item_type": "品牌/品类",
      "context": "用户提及该产品时的场景或评价"
    }}
  ],
  "business_recommendations": [
    {{
      "category": "推荐的细分品类",
      "rationale": "推荐理由（基于评论中的求购信号）",
      "confidence": "高/中/低"
    }}
  ]
}}

输出严格 JSON："""


def _build_comments_text(comments: list[CommentRecord], max_comments: int = 30) -> str:
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


class MarketIntelligenceAnalyst:
    """LLM 3：产品口碑与竞品情报分析师。

    参数：
        llm_client: LLM 客户端实例
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm = llm_client

    def execute(
        self,
        insight: InsightRecord,
        comments: list[CommentRecord],
    ) -> MarketIntelligenceResult:
        """执行商业情报分析。

        参数：
            insight: 洞察记录
            comments: 标准化后的评论列表

        返回：
            MarketIntelligenceResult（结构化字段）
        """
        logger.info("MarketIntelligenceAnalyst: comments=%d", len(comments))

        if not self._llm:
            logger.warning("MarketIntelligenceAnalyst: llm_client 为空，返回空结果")
            return MarketIntelligenceResult()

        pain_points = "、".join(insight.pain_points[:10]) if insight.pain_points else "无"
        complaints = "、".join(insight.complaints[:10]) if insight.complaints else "无"
        solutions = "、".join(insight.solutions[:10]) if insight.solutions else "无"
        market_signals = "、".join(insight.market_signals[:10]) if insight.market_signals else "无"
        comments_text = _build_comments_text(comments)

        prompt = _MARKET_INTEL_PROMPT.format(
            pain_points=pain_points,
            complaints=complaints,
            solutions=solutions,
            market_signals=market_signals,
            comments_data=comments_text,
        )

        try:
            text = self._llm.generate(prompt)
            raw = extract_json_from_text(text)
            if raw is None:
                logger.warning(
                    "MarketIntelligenceAnalyst: LLM 返回非法 JSON，返回空结果。"
                    " 原始内容前 300 字符: %s", text[:300],
                )
                return MarketIntelligenceResult()

            # 解析结构化字段
            pain_point_analysis = [
                PainPointItem(
                    issue=item.get("issue", ""),
                    severity=item.get("severity", ""),
                    user_voice=item.get("user_voice", ""),
                )
                for item in raw.get("pain_point_analysis", [])
                if isinstance(item, dict)
            ]
            competitor_landscape = [
                CompetitorItem(
                    name=item.get("name", ""),
                    item_type=item.get("item_type", ""),
                    context=item.get("context", ""),
                )
                for item in raw.get("competitor_landscape", [])
                if isinstance(item, dict)
            ]
            business_recommendations = [
                BizRecommendation(
                    category=item.get("category", ""),
                    rationale=item.get("rationale", ""),
                    confidence=item.get("confidence", ""),
                )
                for item in raw.get("business_recommendations", [])
                if isinstance(item, dict)
            ]

            result = MarketIntelligenceResult(
                executive_summary=raw.get("executive_summary", ""),
                pain_point_analysis=pain_point_analysis,
                competitor_landscape=competitor_landscape,
                business_recommendations=business_recommendations,
            )

            logger.info(
                "MarketIntelligenceAnalyst: 解析成功 summary=%d字, "
                "pain_points=%d, competitors=%d, recommendations=%d",
                len(result.executive_summary),
                len(result.pain_point_analysis),
                len(result.competitor_landscape),
                len(result.business_recommendations),
            )
            return result

        except Exception as e:
            logger.error("MarketIntelligenceAnalyst generate 失败: %s", e)
            return MarketIntelligenceResult()
