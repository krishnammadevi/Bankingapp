__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import textwrap
from dataclasses import dataclass
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM

# 1. Initialize FastAPI app
app = FastAPI(title="Banking Multi-Agent API", version="1.0")

# 2. Configure Shared LLM
llm = LLM(
    model="gpt-4o-mini",
    base_url=os.environ.get("OPENAI_API_BASE"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_retries=5,
    timeout=60
)

BANKING_POLICY_BASE = {
    "fraud": [
        "Any customer report of an unrecognized transaction must NEVER be auto-approved or reversed by an AI agent.",
        "Suspected fraud always triggers escalation to the Fraud Operations team, regardless of transaction amount.",
        "Do not ask the customer to share full card number, CVV, or OTP in chat.",
        "Advise the customer to freeze/block the card via the app or hotline if they suspect compromise."
    ],
    "transaction": [
        "Failed transactions where funds were debited but not credited are auto-reversed within 3-5 business days per RBI/NPCI guidelines; inform the customer of this timeline.",
        "Duplicate transaction complaints require the last 4 digits of the card/account and approximate date/time before any lookup is quoted.",
        "Never confirm or deny another customer's account details."
    ],
    "loan": [
        "Loan eligibility and interest rate quotes are indicative only; final approval requires a credit bureau check.",
        "Do not disclose a customer's credit score or another party's loan status.",
        "Loan foreclosure/prepayment queries should mention potential foreclosure charges per the loan agreement."
    ],
    "general": [
        "General queries (branch hours, documentation needed, how-to) can be answered directly from the FAQ knowledge base.",
        "Never share other customers' data, even partially, in a general query response."
    ]
}

# 3. Agent Definitions
intent_agent = Agent(
    role="Banking Intent Classification Specialist",
    goal="Classify query into 'fraud', 'transaction', 'loan', or 'general' with confidence score and urgency flags.",
    backstory="Senior triage analyst at a retail bank's contact center.",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

policy_agent = Agent(
    role="Banking Rules & Policy Reasoning Analyst",
    goal="Explain in plain language which bank policy rules apply to the customer message.",
    backstory="Compliance-trained banking operations analyst.",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

drafting_agent = Agent(
    role="Customer Response Drafting Specialist",
    goal="Write a short, warm customer reply grounded strictly in policy analysis.",
    backstory="Customer communications specialist.",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

risk_agent = Agent(
    role="Risk & Escalation Analyst",
    goal="Calculate risk score (0-100) and state reason.",
    backstory="Risk officer deciding auto-resolve vs human escalation.",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

@dataclass
class CaseRecord:
    query_id: str
    raw_message: str
    intent: Optional[str] = None
    intent_confidence: Optional[int] = None
    urgency_flags: Optional[str] = None
    policy_analysis: Optional[str] = None
    draft_response: Optional[str] = None
    risk_score: Optional[int] = None
    risk_reason: Optional[str] = None
    decision: Optional[str] = None
    escalation_target: Optional[str] = None

def _safe_json(raw_text: str, default: dict) -> dict:
    cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            pass
    return default

def process_customer_query(query_id: str, message: str) -> CaseRecord:
    case = CaseRecord(query_id=query_id, raw_message=message)

    # Stage 1: Intent
    t1 = Task(
        description=f'Message: "{message}"\nReturn JSON: {{"intent": "fraud"|"transaction"|"loan"|"general", "confidence": int, "urgency_flags": str}}',
        expected_output="JSON object", agent=intent_agent
    )
    p1 = _safe_json(str(Crew(agents=[intent_agent], tasks=[t1], process=Process.sequential).kickoff()), {"intent": "general", "confidence": 50, "urgency_flags": "none"})
    case.intent = p1.get("intent", "general")
    case.intent_confidence = int(p1.get("confidence", 50))
    case.urgency_flags = p1.get("urgency_flags", "none")

    # Stage 2: Policy Analysis
    rules = BANKING_POLICY_BASE.get(case.intent, BANKING_POLICY_BASE["general"])
    rules_text = "\n".join(f"- {r}" for r in rules)
    t2 = Task(
        description=f'Customer message: "{message}"\nIntent: {case.intent}\nPolicy rules:\n{rules_text}\nExplain which rules apply in 2-3 sentences.',
        expected_output="Policy analysis text", agent=policy_agent
    )
    case.policy_analysis = str(Crew(agents=[policy_agent], tasks=[t2], process=Process.sequential).kickoff()).strip()

    # Stage 3: Draft Response
    t3 = Task(
        description=f'Customer message: "{message}"\nPolicy analysis: "{case.policy_analysis}"\nDraft customer reply under 100 words.',
        expected_output="Customer response", agent=drafting_agent
    )
    case.draft_response = str(Crew(agents=[drafting_agent], tasks=[t3], process=Process.sequential).kickoff()).strip()

    # Stage 4: Risk Assessment
    t4 = Task(
        description=f'Intent: {case.intent}\nUrgency: {case.urgency_flags}\nPolicy analysis: "{case.policy_analysis}"\nDraft: "{case.draft_response}"\nReturn JSON: {{"risk_score": int, "risk_reason": str}}',
        expected_output="JSON object", agent=risk_agent
    )
    p4 = _safe_json(str(Crew(agents=[risk_agent], tasks=[t4], process=Process.sequential).kickoff()), {"risk_score": 50, "risk_reason": "Fallback"})
    case.risk_score = int(p4.get("risk_score", 50))
    case.risk_reason = p4.get("risk_reason", "")

    # Escalation Rule
    if case.intent == "fraud" or case.risk_score >= 60 or case.intent_confidence < 50:
        case.decision = "ESCALATE"
        case.escalation_target = "Fraud Operations" if case.intent == "fraud" else "Human Support Agent"
    else:
        case.decision = "AUTO_RESOLVE"
        case.escalation_target = None

    return case

class BankingQueryRequest(BaseModel):
    query_id: str
    message: str

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Banking Multi-Agent Pipeline"}

@app.post("/process_query")
def process_query(request: BankingQueryRequest):
    try:
        case = process_customer_query(request.query_id, request.message)
        return {
            "query_id": case.query_id,
            "intent": case.intent,
            "confidence": case.intent_confidence,
            "policy_analysis": case.policy_analysis,
            "response": case.draft_response,
            "risk_score": case.risk_score,
            "risk_reason": case.risk_reason,
            "decision": case.decision,
            "escalation_target": case.escalation_target
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
