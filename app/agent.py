import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    APPOINTMENT_BOOKING = "appointment_booking"
    MEDICATION_REFILL = "medication_refill"
    DOCUMENT_QA = "document_qa"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


@dataclass
class AgentResponse:
    intent: QueryIntent
    tool_used: Optional[str]
    tool_result: Optional[Dict[str, Any]]
    routed_to_rag: bool
    routing_reason: str


# ─────────────────────────────────────────────
#  Keyword-based intent classifier
# ─────────────────────────────────────────────

INTENT_KEYWORDS: Dict[QueryIntent, List[str]] = {
    QueryIntent.EMERGENCY: [
        "chest pain", "can't breathe", "difficulty breathing", "heart attack",
        "stroke", "unconscious", "seizure", "severe bleeding", "911", "emergency",
        "dying", "not breathing",
    ],
    QueryIntent.APPOINTMENT_BOOKING: [
        "book", "schedule", "appointment", "slot", "available", "availability",
        "when can i", "see a doctor", "visit", "reserve", "cancel appointment",
        "reschedule", "next available",
    ],
    QueryIntent.MEDICATION_REFILL: [
        "refill", "prescription", "running out", "out of medication",
        "need more pills", "renew prescription",
    ],
}


def classify_intent(question: str) -> QueryIntent:
    """
    Simple keyword-based intent router.
    Returns the most specific intent match, falling back to DOCUMENT_QA.
    """
    question_lower = question.lower()

    # Check emergency first — highest priority
    for intent in [QueryIntent.EMERGENCY, QueryIntent.APPOINTMENT_BOOKING, QueryIntent.MEDICATION_REFILL]:
        for keyword in INTENT_KEYWORDS.get(intent, []):
            if keyword in question_lower:
                logger.info(f"Intent '{intent}' matched on keyword '{keyword}'")
                return intent

    return QueryIntent.DOCUMENT_QA


# ─────────────────────────────────────────────
#  Mock Tool: check_available_slots
# ─────────────────────────────────────────────

DEPARTMENTS = [
    "Cardiology", "Orthopedics", "Neurology", "Dermatology",
    "Gastroenterology", "Endocrinology", "Pediatrics", "Primary Care",
    "OB/GYN", "Psychiatry and Behavioral Health",
]

TIME_SLOTS = ["9:00 AM", "10:30 AM", "11:00 AM", "2:00 PM", "3:30 PM", "4:00 PM", "5:15 PM"]


def check_available_slots(department: str, requested_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Mock appointment availability tool.
    Simulates querying a scheduling system and returns available time slots.
    """
    # Normalize department
    matched_dept = next(
        (d for d in DEPARTMENTS if department.lower() in d.lower()),
        department.title(),
    )

    # Generate mock upcoming dates
    today = datetime.today()
    available_dates = []
    for offset in range(1, 10):
        candidate = today + timedelta(days=offset)
        if candidate.weekday() < 6:  # Mon–Sat
            available_dates.append(candidate)
        if len(available_dates) == 3:
            break

    # Build mock slots
    slots = []
    for date in available_dates:
        # Randomly assign 2–4 available slots per day
        day_slots = random.sample(TIME_SLOTS, k=random.randint(2, 4))
        for slot_time in sorted(day_slots):
            slots.append({
                "date": date.strftime("%A, %B %d, %Y"),
                "time": slot_time,
                "provider": f"Dr. {random.choice(['Sharma', 'Patel', 'Singh', 'Mehta', 'Gupta'])}",
                "type": "In-Person",
                "location": "Main Clinic — Building A",
            })

    logger.info(f"check_available_slots called: dept='{matched_dept}', found {len(slots)} slots")

    return {
        "department": matched_dept,
        "requested_date": requested_date or "Next available",
        "available_slots": slots[:6],  # Return max 6 slots
        "booking_url": "https://portal.healthcarefacility.org/book",
        "phone": "1-800-APPT-NOW",
    }


def check_refill_status(medication_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Mock tool to check medication refill eligibility status.
    """
    statuses = ["Eligible for refill", "Requires lab work before refill", "Requires in-person visit"]
    result = {
        "medication": medication_name or "Your medication",
        "status": random.choice(statuses),
        "last_filled": (datetime.today() - timedelta(days=random.randint(20, 80))).strftime("%B %d, %Y"),
        "refills_remaining": random.randint(0, 5),
        "portal_url": "https://portal.healthcarefacility.org/medications",
        "pharmacy_phone": "1-800-MED-REFILL",
    }
    logger.info(f"check_refill_status called for '{medication_name}': {result['status']}")
    return result


# ─────────────────────────────────────────────
#  Agent Router
# ─────────────────────────────────────────────

def extract_department(question: str) -> str:
    """Best-effort department extraction from question text."""
    q = question.lower()
    for dept in DEPARTMENTS:
        if dept.lower() in q or dept.lower().split()[0] in q:
            return dept
    return "General / Primary Care"


def extract_medication_name(question: str) -> Optional[str]:
    """Best-effort medication name extraction."""
    words = question.split()
    for i, word in enumerate(words):
        if word.lower() in ("for", "my", "the") and i + 1 < len(words):
            return words[i + 1].strip(".,?")
    return None


def route_and_run(question: str) -> AgentResponse:
    """
    Main agent entry point.
    Classifies intent, runs the appropriate tool, returns structured response.
    For DOCUMENT_QA and tool-fallback, signals to call the RAG pipeline.
    """
    intent = classify_intent(question)

    if intent == QueryIntent.EMERGENCY:
        return AgentResponse(
            intent=intent,
            tool_used=None,
            tool_result={
                "message": (
                    "⚠️ This appears to be a medical emergency. "
                    "Please call 911 immediately or go to the nearest emergency room. "
                    "Do not wait for an online response."
                )
            },
            routed_to_rag=False,
            routing_reason="Emergency detected — bypass RAG, show urgent message.",
        )

    elif intent == QueryIntent.APPOINTMENT_BOOKING:
        department = extract_department(question)
        tool_result = check_available_slots(department=department)
        return AgentResponse(
            intent=intent,
            tool_used="check_available_slots",
            tool_result=tool_result,
            routed_to_rag=True,  # Also pull policy info from RAG
            routing_reason=f"Appointment booking request detected. Running mock scheduling tool for '{department}'.",
        )

    elif intent == QueryIntent.MEDICATION_REFILL:
        med_name = extract_medication_name(question)
        tool_result = check_refill_status(medication_name=med_name)
        return AgentResponse(
            intent=intent,
            tool_used="check_refill_status",
            tool_result=tool_result,
            routed_to_rag=True,  # Also pull refill policy from RAG
            routing_reason="Medication refill request. Running mock refill status tool.",
        )

    else:
        return AgentResponse(
            intent=QueryIntent.DOCUMENT_QA,
            tool_used=None,
            tool_result=None,
            routed_to_rag=True,
            routing_reason="General question — routing to RAG pipeline.",
        )
