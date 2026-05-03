from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Magicpin Merchant AI Assistant")
START_TIME = time.time()

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}
MAX_ACTIONS_PER_TICK = 5

contexts: Dict[tuple[str, str], Dict[str, Any]] = {}
conversations: Dict[str, List[Dict[str, Any]]] = {}
conversation_meta: Dict[str, Dict[str, Any]] = {}
dispatched_triggers: Dict[str, str] = {}


class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


def _iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _get_payload(scope: str, context_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not context_id:
        return None
    entry = contexts.get((scope, context_id))
    if not entry:
        return None
    return entry.get("payload")


def _counts_by_scope() -> Dict[str, int]:
    counts: Dict[str, int] = {s: 0 for s in VALID_SCOPES}
    for (scope, _), _value in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return counts


def _percent(value: Optional[float], digits: int = 1, show_sign: bool = False) -> str:
    if value is None:
        return "--"
    fmt = f"{{:+.{digits}f}}%" if show_sign else f"{{:.{digits}f}}%"
    return fmt.format(value * 100)


def _choose_offer(merchant: Dict[str, Any], category: Dict[str, Any]) -> str:
    merchant_offers = merchant.get("offers", [])
    active_offer = next((o for o in merchant_offers if o.get("status") == "active"), None)
    if active_offer:
        return active_offer.get("title", "custom offer")
    catalog = category.get("offer_catalog", [])
    if catalog:
        first = catalog[0]
        if isinstance(first, dict):
            return first.get("title", "category offer")
        return str(first)
    return f"{category.get('slug', 'listing')} offer"


def _has_hinglish_pref(merchant: Dict[str, Any]) -> bool:
    langs = [lang.lower() for lang in merchant.get("identity", {}).get("languages", [])]
    return any(lang.startswith("hi") for lang in langs)


def _closing_line(merchant: Dict[str, Any]) -> str:
    if _has_hinglish_pref(merchant):
        return "YES likho aur main turant draft bhejti hoon. STOP likh ke pause kar sakte ho."
    return "Reply YES and I'll draft it now. Send STOP to pause."


def _find_digest_item(category: Dict[str, Any], trigger_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    target_id = trigger_payload.get("top_item_id")
    digest_items = category.get("digest", [])
    if target_id:
        for item in digest_items:
            if item.get("id") == target_id:
                return item
    return digest_items[0] if digest_items else None


def _compose_research_digest(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> Dict[str, Any]:
    digest_item = _find_digest_item(category, trigger.get("payload", {}))
    merchant_name = merchant.get("identity", {}).get("name", "there")
    locality = merchant.get("identity", {}).get("locality", merchant.get("identity", {}).get("city", "your area"))
    peer_ctr = category.get("peer_stats", {}).get("avg_ctr")
    merchant_ctr = merchant.get("performance", {}).get("ctr")
    offer_title = _choose_offer(merchant, category)
    digest_title = digest_item.get("title") if digest_item else "category update"
    digest_source = digest_item.get("source") if digest_item else category.get("slug", "category")
    trial_size = digest_item.get("trial_n") if digest_item else None
    patient_segment = digest_item.get("patient_segment") if digest_item else None

    body_parts = [
        f"{merchant_name}, {digest_title} — {digest_source}.",
    ]
    if trial_size:
        body_parts.append(f"Study size {trial_size} patients, segment {patient_segment or 'core cohort'}.")
    if merchant_ctr and peer_ctr:
        body_parts.append(
            f"Your CTR is {_percent(merchant_ctr)} vs peer {_percent(peer_ctr)} in {locality}."
        )
    body_parts.append(
        f"Want me to pull the abstract + draft a WhatsApp around {offer_title}? {_closing_line(merchant)}"
    )
    body = " ".join(body_parts)
    return {
        "body": body,
        "cta": "binary_yes_stop",
        "send_as": "vera",
        "template_name": "vera_research_digest_v1",
        "template_params": [merchant_name, digest_title, digest_source],
        "suppression_key": trigger.get("suppression_key") or f"{trigger.get('kind')}: {trigger.get('id')}",
        "rationale": f"Research digest trigger referencing {digest_title} for {merchant_name}",
        "meta": {
            "fact": digest_title,
            "kind": "research_digest",
        },
    }


def _compose_perf_health(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any], mode: str) -> Dict[str, Any]:
    merchant_name = merchant.get("identity", {}).get("name", "there")
    locality = merchant.get("identity", {}).get("locality", merchant.get("identity", {}).get("city", "your area"))
    perf = merchant.get("performance", {})
    views = perf.get("views")
    calls = perf.get("calls")
    ctr = perf.get("ctr")
    delta_views = perf.get("delta_7d", {}).get("views_pct")
    peer_ctr = category.get("peer_stats", {}).get("avg_ctr")
    offer = _choose_offer(merchant, category)
    direction = "up" if mode == "spike" else "down"

    body = (
        f"{merchant_name}, {_percent(delta_views, digits=0, show_sign=True)} {direction} shift on searches "
        f"({views} views / {calls} calls) for your {category.get('slug', '')} listing in {locality}. "
        f"CTR {_percent(ctr)} vs peer {_percent(peer_ctr)}. Shall I package {offer} into a fresh Google Post? {_closing_line(merchant)}"
    )

    template = f"vera_perf_{mode}_v1"
    return {
        "body": body,
        "cta": "binary_yes_stop",
        "send_as": "vera",
        "template_name": template,
        "template_params": [merchant_name, f"{views}", _percent(delta_views, digits=0, show_sign=True)],
        "suppression_key": trigger.get("suppression_key") or f"{trigger.get('kind')}: {trigger.get('id')}",
        "rationale": f"Performance {direction} trigger with concrete stats",
        "meta": {
            "kind": trigger.get("kind"),
            "fact": f"{views} views / {calls} calls",
        },
    }


def _compose_generic(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> Dict[str, Any]:
    merchant_name = merchant.get("identity", {}).get("name", "there")
    city = merchant.get("identity", {}).get("city", "your city")
    offer = _choose_offer(merchant, category)
    trigger_kind = trigger.get("kind", "insight")
    body = (
        f"{merchant_name}, quick {trigger_kind.replace('_', ' ')} ping for your {category.get('slug', '')} listing in {city}. "
        f"Ready to spotlight {offer}? {_closing_line(merchant)}"
    )
    return {
        "body": body,
        "cta": "binary_yes_stop",
        "send_as": "vera",
        "template_name": f"vera_{trigger_kind}_v1",
        "template_params": [merchant_name, offer, trigger_kind],
        "suppression_key": trigger.get("suppression_key") or f"{trigger.get('kind')}: {trigger.get('id')}",
        "rationale": f"Default template for {trigger_kind}",
        "meta": {
            "kind": trigger_kind,
            "fact": offer,
        },
    }


def _compose_customer(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any], customer: Dict[str, Any]) -> Dict[str, Any]:
    customer_name = customer.get("identity", {}).get("name", "there")
    merchant_name = merchant.get("identity", {}).get("name", "")
    offer = _choose_offer(merchant, category)
    last_visit = customer.get("relationship", {}).get("last_visit", "recently")
    body = (
        f"Hi {customer_name}, {merchant_name} yahan. {trigger.get('kind', 'update')} reminder — last visit {last_visit}. "
        f"Slot hold karun for {offer}? Reply 1 for YES, 2 for another time."
    )
    return {
        "body": body,
        "cta": "multi_choice",
        "send_as": "merchant_on_behalf",
        "template_name": "merchant_customer_followup_v1",
        "template_params": [customer_name, offer, last_visit],
        "suppression_key": trigger.get("suppression_key") or f"{trigger.get('kind')}: {trigger.get('id')}",
        "rationale": "Customer follow-up drafted on behalf of merchant",
        "meta": {
            "kind": trigger.get("kind"),
            "fact": last_visit,
        },
    }


def compose_message(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any], customer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if customer:
        return _compose_customer(category, merchant, trigger, customer)

    kind = trigger.get("kind")
    if kind == "research_digest":
        return _compose_research_digest(category, merchant, trigger)
    if kind == "perf_spike":
        return _compose_perf_health(category, merchant, trigger, mode="spike")
    if kind == "perf_dip":
        return _compose_perf_health(category, merchant, trigger, mode="dip")
    return _compose_generic(category, merchant, trigger)


def _build_action(trigger_id: str) -> Optional[Dict[str, Any]]:
    trigger_payload = _get_payload("trigger", trigger_id)
    if not trigger_payload:
        return None
    merchant = _get_payload("merchant", trigger_payload.get("merchant_id"))
    if not merchant:
        return None
    category = _get_payload("category", merchant.get("category_slug"))
    if not category:
        return None
    customer = _get_payload("customer", trigger_payload.get("customer_id"))

    composed = compose_message(category, merchant, trigger_payload, customer)
    meta = composed.pop("meta", {})
    conversation_id = f"conv_{trigger_payload.get('merchant_id')}_{trigger_id}"
    action = {
        "conversation_id": conversation_id,
        "merchant_id": trigger_payload.get("merchant_id"),
        "customer_id": trigger_payload.get("customer_id"),
        "send_as": composed["send_as"],
        "trigger_id": trigger_id,
        "template_name": composed["template_name"],
        "template_params": composed["template_params"],
        "body": composed["body"],
        "cta": composed["cta"],
        "suppression_key": composed["suppression_key"],
        "rationale": composed["rationale"],
    }
    conversation_meta[conversation_id] = {
        "merchant_id": trigger_payload.get("merchant_id"),
        "customer_id": trigger_payload.get("customer_id"),
        "trigger_id": trigger_id,
        "trigger_kind": trigger_payload.get("kind"),
        "context_fact": meta.get("fact"),
    }
    return action


@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": _counts_by_scope(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Cascade Sprint",
        "team_members": ["Cascade"],
        "model": "deterministic-template",
        "approach": "Rule-based first flow using category+merchant+trigger context",
        "contact_email": "bot@example.com",
        "version": "0.1.0",
        "submitted_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in VALID_SCOPES:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": body.scope},
        )
    key = (body.scope, body.context_id)
    existing = contexts.get(key)
    if existing and existing.get("version", 0) >= body.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": existing.get("version", 0)},
        )
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": _iso_now(),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions: List[Dict[str, Any]] = []
    for trigger_id in sorted(dict.fromkeys(body.available_triggers)):
        if trigger_id in dispatched_triggers:
            continue
        action = _build_action(trigger_id)
        if action:
            dispatched_triggers[trigger_id] = action["conversation_id"]
            actions.append(action)
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break
    return {"actions": actions}


def _craft_followup(state: Optional[Dict[str, Any]], merchant_message: str) -> Dict[str, Any]:
    normalized = merchant_message.strip().lower()
    trigger_kind = state.get("trigger_kind") if state else "generic"
    fact = state.get("context_fact") if state else "the insight"
    merchant = _get_payload("merchant", state.get("merchant_id")) if state else None
    closing = _closing_line(merchant) if merchant else "Reply YES to proceed."

    if any(word in normalized for word in ["yes", "ok", "sure", "do it", "send"]):
        body = f"Great, actioning the {trigger_kind.replace('_', ' ')} insight around {fact}. I'll update you once shipped."
        return {"body": body, "cta": "open_ended", "rationale": "Merchant accepted"}
    if "stop" in normalized or "no" in normalized:
        return {"body": "Noted. Pausing this thread. Ping me when you want to restart.", "cta": "none", "rationale": "Merchant declined"}
    body = f"Sharing more detail on {fact}. Anything specific you want me to focus on? {closing}"
    return {"body": body, "cta": "open_ended", "rationale": "Clarifying follow-up"}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conversations.setdefault(body.conversation_id, []).append(
        {"from": body.from_role, "message": body.message, "received_at": body.received_at}
    )
    state = conversation_meta.get(body.conversation_id)
    normalized = body.message.strip().lower()
    if "stop" in normalized:
        return {"action": "end", "rationale": "Merchant asked to stop"}
    if any(word in normalized for word in ["later", "busy", "wait"]):
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Merchant requested time"}

    followup = _craft_followup(state, body.message)
    return {
        "action": "send",
        "body": followup["body"],
        "cta": followup["cta"],
        "rationale": followup["rationale"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bot:app", host="0.0.0.0", port=8080, reload=False)
