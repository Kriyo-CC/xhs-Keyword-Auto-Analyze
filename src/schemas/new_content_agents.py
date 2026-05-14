"""新版内容 Agent 输出 schema。

包含三个 LLM Agent 的输入输出数据结构：
- IdeationResult / TopicIdea：LLM 1 爆款选题与内容策划专家
- CopywritingResult / PainPointSolution / CopySection：LLM 2 小红书"网感"文案写手
- MarketIntelligenceResult / PainPointItem / CompetitorItem / BizRecommendation：LLM 3 产品口碑与竞品情报分析师

LLM 2 和 LLM 3 均输出结构化 JSON，由报告渲染器转换为 HTML 卡片展示，
不再以 Markdown 或纯文本形式输出。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopicIdea(BaseModel):
    """LLM 1 生成的单条爆款选题。

    包含爆款标题、目标人群、核心抓手和内容大纲。
    """

    topic_title: str = ""
    target_audience: str = ""
    core_hook: str = ""
    content_framework: str = ""


class IdeationResult(BaseModel):
    """LLM 1 爆款选题与内容策划专家的最终输出。

    包含 3-5 个具有极高爆款潜力的新选题。
    """

    topic_ideas: list[TopicIdea] = Field(default_factory=list)


# ============================================================================
# LLM 2 结构化输出
# ============================================================================


class PainPointSolution(BaseModel):
    """单个痛点-方案对。"""

    pain_point: str = ""
    solution: str = ""


class CopySection(BaseModel):
    """文案正文中的一个小节。"""

    heading: str = ""
    content: str = ""


class CopywritingResult(BaseModel):
    """LLM 2 小红书"网感"文案写手的最终输出（结构化 JSON）。

    字段说明：
    - title: 爆款标题（含 emoji，双标题或悬念式）
    - opening_hook: 引入钩子，使用真实用户评论拉近距离
    - pain_point_solutions: 痛点-解决方案对照列表
    - dry_goods: 干货要点（短句列表）
    - body_sections: 正文小节（可选，用于灵活分段）
    - call_to_action: 文末互动钩子
    - hashtags: 话题标签列表（3-5 个，不含 # 前缀）
    - selected_topic_title: 对应 LLM 1 的选题标题
    """

    title: str = ""
    opening_hook: str = ""
    pain_point_solutions: list[PainPointSolution] = Field(default_factory=list)
    dry_goods: list[str] = Field(default_factory=list)
    body_sections: list[CopySection] = Field(default_factory=list)
    call_to_action: str = ""
    hashtags: list[str] = Field(default_factory=list)
    selected_topic_title: str = ""

    @property
    def has_content(self) -> bool:
        """是否存在有效内容（至少有一个非空字段）。"""
        return bool(
            self.title
            or self.opening_hook
            or self.pain_point_solutions
            or self.dry_goods
            or self.body_sections
            or self.call_to_action
        )


# ============================================================================
# LLM 3 结构化输出
# ============================================================================


class PainPointItem(BaseModel):
    """单个产品痛点条目。"""

    issue: str = ""
    severity: str = ""  # 高 / 中 / 低
    user_voice: str = ""  # 用户原声证据


class CompetitorItem(BaseModel):
    """单个竞品/替代方案条目。"""

    name: str = ""
    item_type: str = ""  # 品牌 / 品类
    context: str = ""  # 用户提及场景


class BizRecommendation(BaseModel):
    """单条商业选品建议。"""

    category: str = ""
    rationale: str = ""
    confidence: str = ""  # 高 / 中 / 低


class MarketIntelligenceResult(BaseModel):
    """LLM 3 产品口碑与竞品情报分析师的最终输出（结构化 JSON）。

    字段说明：
    - executive_summary: 报告摘要（2-4 句话）
    - pain_point_analysis: 产品痛点分析列表
    - competitor_landscape: 热门替代品/竞品列表
    - business_recommendations: 商业变现/选品建议列表（推荐 3 个细分品类）
    """

    executive_summary: str = ""
    pain_point_analysis: list[PainPointItem] = Field(default_factory=list)
    competitor_landscape: list[CompetitorItem] = Field(default_factory=list)
    business_recommendations: list[BizRecommendation] = Field(default_factory=list)

    @property
    def has_content(self) -> bool:
        """是否存在有效内容。"""
        return bool(
            self.executive_summary
            or self.pain_point_analysis
            or self.competitor_landscape
            or self.business_recommendations
        )
