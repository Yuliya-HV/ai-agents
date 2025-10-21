"""
Agent-Based Classification with Structured JSON Output - LLM Gemini
"""

import os
from dotenv import load_dotenv
from google import genai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    CLASSIFIER = "classifier"
    VALIDATOR = "validator"
    CONFIDENCE_ASSESSOR = "confidence_assessor"
    ESCALATION_MANAGER = "escalation_manager"


@dataclass
class AgentResponse:
    role: AgentRole
    content: Any
    confidence: float
    reasoning: str


class GeminiStructuredAgent:
    """Base agent using Gemini with JSON schema definitions."""

    def __init__(self, role: AgentRole, client, model_name: str = "gemini-2.0-flash-exp"):
        self.role = role
        self.client = client
        self.model_name = model_name

    def _call_gemini_structured(self, system_prompt: str, user_prompt: str, response_schema: Dict[str, Any]) -> Dict[
        str, Any]:
        """Call Gemini with JSON schema definition."""
        try:
            # Combine prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Add schema instruction to the prompt
            schema_instruction = f"""
            Return your response as a valid JSON object that matches this schema:
            {json.dumps(response_schema, indent=2)}

            Return ONLY the JSON object, no other text.
            """

            full_prompt += schema_instruction

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )

            # Parse the JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            return json.loads(response_text)

        except Exception as e:
            logger.error(f"Gemini API call failed for {self.role.value}: {e}")
            raise


class ClassifierAgent(GeminiStructuredAgent):
    """Classifier agent using JSON schema."""

    def classify(self, message: str, labels: List[str], context: Dict[str, Any] = None) -> AgentResponse:
        """Classify message with JSON output."""

        system_prompt = self._build_system_prompt(labels, context)
        user_prompt = f"Classify this message: {message}"

        response_schema = {
            "type": "object",
            "properties": {
                "assigned_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of applicable labels"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation for label assignments"
                },
                "confidence": {
                    "type": "number",
                    "description": "Overall confidence in classification (0.0-1.0)"
                }
            },
            "required": ["assigned_labels", "reasoning", "confidence"]
        }

        try:
            result = self._call_gemini_structured(system_prompt, user_prompt, response_schema)

            # Validate labels are in our schema
            valid_labels = [label for label in result.get('assigned_labels', []) if label in labels]

            return AgentResponse(
                role=self.role,
                content=valid_labels,
                confidence=result.get('confidence', 0.5),
                reasoning=result.get('reasoning', 'No reasoning provided')
            )

        except Exception as e:
            logger.error(f"Classifier failed: {e}")
            return AgentResponse(
                role=self.role,
                content=[],
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}"
            )

    def _build_system_prompt(self, labels: List[str], context: Dict[str, Any]) -> str:
        context_str = ""
        if context and context.get('conversation_history'):
            context_str = f"\nConversation Context: {context['conversation_history']}"

        return f"""
        You are an expert message classifier. Analyze the user message and identify ALL applicable labels.

        AVAILABLE LABELS: {', '.join(labels)}
        {context_str}

        CLASSIFICATION GUIDELINES:
        1. Be precise - only assign labels that clearly apply
        2. Consider both explicit and implicit meanings
        3. Assign multiple labels when appropriate
        4. Consider conversation context and user intent
        5. Be conservative - when in doubt, don't assign
        """


class ValidatorAgent(GeminiStructuredAgent):
    """Validator agent using JSON schema."""

    def validate(self, message: str, proposed_labels: List[str], classifier_reasoning: str,
                 labels: List[str]) -> AgentResponse:
        """Validate classifications with JSON output."""

        system_prompt = f"""
        You are a classification validator. Review the proposed labels and reasoning.

        AVAILABLE LABELS: {', '.join(labels)}

        Your Task:
        1. Verify each proposed label is appropriate
        2. Suggest any missing labels from the available list
        3. Remove any incorrect labels
        4. Provide validation notes explaining changes
        """

        user_prompt = f"""
        Message: {message}
        Proposed Labels: {proposed_labels}
        Classifier Reasoning: {classifier_reasoning}
        """

        response_schema = {
            "type": "object",
            "properties": {
                "validated_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Final list of validated labels"
                },
                "validation_notes": {
                    "type": "string",
                    "description": "Assessment and changes made"
                },
                "validation_confidence": {
                    "type": "number",
                    "description": "Confidence in validation (0.0-1.0)"
                }
            },
            "required": ["validated_labels", "validation_notes", "validation_confidence"]
        }

        try:
            result = self._call_gemini_structured(system_prompt, user_prompt, response_schema)

            valid_labels = [label for label in result.get('validated_labels', []) if label in labels]

            return AgentResponse(
                role=self.role,
                content=valid_labels,
                confidence=result.get('validation_confidence', 0.5),
                reasoning=result.get('validation_notes', 'No validation notes provided')
            )

        except Exception as e:
            logger.error(f"Validator failed: {e}")
            return AgentResponse(
                role=self.role,
                content=proposed_labels,
                confidence=0.3,
                reasoning=f"Validation failed: {str(e)}"
            )


class ConfidenceAssessorAgent(GeminiStructuredAgent):
    """Assess confidence scores using JSON schema."""

    def assess_confidence(self, message: str, final_labels: List[str],
                          classifier_reasoning: str, validator_notes: str) -> AgentResponse:
        """Assess individual confidence scores."""

        system_prompt = """
        You are a confidence assessor. Assign individual confidence scores for each label.

        Considerations:
        - Explicit vs implicit evidence in the message
        - Strength of supporting reasoning
        - Ambiguity or multiple interpretations possible

        Scoring Guidelines:
        - 0.9-1.0: Very strong, unambiguous evidence
        - 0.7-0.8: Strong evidence, minor ambiguity
        - 0.5-0.6: Moderate evidence, some ambiguity
        - 0.3-0.4: Weak evidence, significant ambiguity
        - 0.0-0.2: Very weak evidence, high ambiguity
        """

        user_prompt = f"""
        Message: {message}
        Final Labels: {final_labels}
        Classifier Reasoning: {classifier_reasoning}
        Validator Notes: {validator_notes}
        """

        # Create a simpler schema without additionalProperties
        properties = {}
        for label in final_labels:
            properties[label] = {
                "type": "number",
                "description": f"Confidence score for {label} (0.0-1.0)"
            }

        response_schema = {
            "type": "object",
            "properties": {
                "confidence_scores": {
                    "type": "object",
                    "properties": properties
                },
                "assessment_notes": {
                    "type": "string",
                    "description": "Brief explanation of scores"
                },
                "overall_confidence": {
                    "type": "number",
                    "description": "Your confidence in these assessments (0.0-1.0)"
                }
            },
            "required": ["confidence_scores", "assessment_notes", "overall_confidence"]
        }

        try:
            result = self._call_gemini_structured(system_prompt, user_prompt, response_schema)

            # Extract confidence scores
            confidence_scores = result.get('confidence_scores', {})
            validated_scores = {}
            for label in final_labels:
                score = confidence_scores.get(label, 0.5)
                validated_scores[label] = max(0.0, min(1.0, float(score)))

            return AgentResponse(
                role=self.role,
                content=validated_scores,
                confidence=result.get('overall_confidence', 0.5),
                reasoning=result.get('assessment_notes', 'No assessment notes provided')
            )

        except Exception as e:
            logger.error(f"Confidence assessment failed: {e}")
            fallback_scores = {label: 0.7 for label in final_labels}
            return AgentResponse(
                role=self.role,
                content=fallback_scores,
                confidence=0.5,
                reasoning=f"Confidence assessment failed: {str(e)}"
            )


class EscalationManagerAgent(GeminiStructuredAgent):
    """Escalation manager using JSON schema."""

    def assess_escalation(self, message: str, labels: List[str],
                          confidence_scores: Dict[str, float], all_reasoning: Dict[str, str]) -> AgentResponse:
        """Determine if human review is needed."""

        system_prompt = """
        You are an escalation manager. Decide if this classification needs human review.

        Escalation Criteria (escalate if ANY are true):
        - Any confidence score < 0.6
        - High-risk labels present (complaint, legal, emergency, security)
        - Contradictory reasoning between agents
        - Message contains urgent language (immediately, emergency, ASAP)
        - Low overall confidence in the classification
        """

        user_prompt = f"""
        Message: {message}
        Assigned Labels: {labels}
        Confidence Scores: {confidence_scores}
        Classifier Reasoning: {all_reasoning.get('classifier', 'None')}
        Validator Reasoning: {all_reasoning.get('validator', 'None')}
        """

        response_schema = {
            "type": "object",
            "properties": {
                "needs_human_review": {
                    "type": "boolean",
                    "description": "Whether human review is required"
                },
                "escalation_reason": {
                    "type": "string",
                    "description": "Explanation for escalation decision"
                },
                "escalation_confidence": {
                    "type": "number",
                    "description": "Confidence in escalation decision (0.0-1.0)"
                }
            },
            "required": ["needs_human_review", "escalation_reason", "escalation_confidence"]
        }

        try:
            result = self._call_gemini_structured(system_prompt, user_prompt, response_schema)

            return AgentResponse(
                role=self.role,
                content=result.get('needs_human_review', True),  # Default to True for safety
                confidence=result.get('escalation_confidence', 0.5),
                reasoning=result.get('escalation_reason', 'No escalation reason provided')
            )

        except Exception as e:
            logger.error(f"Escalation assessment failed: {e}")
            return AgentResponse(
                role=self.role,
                content=True,
                confidence=0.5,
                reasoning=f"Escalation assessment failed: {str(e)}"
            )


class GeminiAgentOrchestrator:
    """Orchestrator using Gemini API with JSON schema."""

    def __init__(self, api_key: str, label_schema: Dict[str, Any], model_name: str = "gemini-2.0-flash-exp"):
        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.labels = label_schema['labels']

        # Initialize agents
        self.agents = {
            AgentRole.CLASSIFIER: ClassifierAgent(AgentRole.CLASSIFIER, self.client, model_name),
            AgentRole.VALIDATOR: ValidatorAgent(AgentRole.VALIDATOR, self.client, model_name),
            AgentRole.CONFIDENCE_ASSESSOR: ConfidenceAssessorAgent(AgentRole.CONFIDENCE_ASSESSOR, self.client,
                                                                   model_name),
            AgentRole.ESCALATION_MANAGER: EscalationManagerAgent(AgentRole.ESCALATION_MANAGER, self.client, model_name),
        }

        logger.info(f"Initialized Gemini agent orchestrator for {len(self.labels)} labels")

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-agent classification pipeline with Gemini."""
        try:
            message = request['message']
            context = request.get('context', {})

            logger.info(f"Starting Gemini agent classification: {message[:100]}...")

            # Step 1: Initial Classification
            classifier_response = self.agents[AgentRole.CLASSIFIER].classify(
                message, self.labels, context
            )
            logger.info(
                f"Classifier: {len(classifier_response.content)} labels, confidence {classifier_response.confidence:.2f}")

            # Step 2: Validation
            validator_response = self.agents[AgentRole.VALIDATOR].validate(
                message, classifier_response.content, classifier_response.reasoning, self.labels
            )
            logger.info(
                f"Validator: {len(validator_response.content)} labels, confidence {validator_response.confidence:.2f}")

            # Step 3: Confidence Assessment
            confidence_response = self.agents[AgentRole.CONFIDENCE_ASSESSOR].assess_confidence(
                message, validator_response.content, classifier_response.reasoning, validator_response.reasoning
            )
            logger.info(f"Confidence Assessor: assessed {len(confidence_response.content)} labels")

            # Step 4: Escalation Decision
            all_reasoning = {
                'classifier': classifier_response.reasoning,
                'validator': validator_response.reasoning
            }
            escalation_response = self.agents[AgentRole.ESCALATION_MANAGER].assess_escalation(
                message, validator_response.content, confidence_response.content, all_reasoning
            )
            logger.info(f"Escalation Manager: human_review={escalation_response.content}")

            # Prepare final response
            final_labels = validator_response.content
            confidence_scores = [confidence_response.content.get(label, 0.5) for label in final_labels]

            # Sort by confidence
            if final_labels and confidence_scores:
                sorted_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=True)
                final_labels = [final_labels[i] for i in sorted_indices]
                confidence_scores = [confidence_scores[i] for i in sorted_indices]

            return {
                'predicted_labels': final_labels,
                'confidence_scores': confidence_scores,
                'needs_human_review': escalation_response.content,
                'agent_reasoning': {
                    'classifier': classifier_response.reasoning,
                    'validator': validator_response.reasoning,
                    'confidence_assessor': confidence_response.reasoning,
                    'escalation_manager': escalation_response.reasoning
                },
                'agent_confidences': {
                    'classifier': classifier_response.confidence,
                    'validator': validator_response.confidence,
                    'confidence_assessor': confidence_response.confidence,
                    'escalation_manager': escalation_response.confidence
                },
                'model_used': self.model_name,
                'success': True
            }

        except Exception as e:
            logger.error(f"Gemini agent pipeline failed: {e}")
            return {
                'predicted_labels': [],
                'confidence_scores': [],
                'needs_human_review': True,
                'error': str(e),
                'success': False
            }


def main():
    """Example usage of the Gemini-based classification system."""

    # Make sure API key is set
    if not os.getenv('GOOGLE_API_KEY'):
        print("Please set GOOGLE_API_KEY environment variable")
        return

    # Label schema
    label_schema = {
        'labels': [
            'greeting', 'question', 'complaint', 'billing_help',
            'technical_support', 'feedback', 'refund_request', 'urgent'
        ]
    }

    # Initialize orchestrator
    orchestrator = GeminiAgentOrchestrator(os.getenv('GOOGLE_API_KEY'), label_schema)

    # Test cases
    test_cases = [
        {
            'message': "Hello, I need urgent help with my billing statement from last month",
            'context': {}
        },
        {
            'message': "This app is completely broken and I want my money back immediately!",
            'context': {}
        },
        {
            'message': "How do I upgrade my plan?",
            'context': {}
        }
    ]

    print("Testing Gemini Agent Classification")
    print("=" * 50)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: '{test_case['message']}'")
        print("-" * 40)

        result = orchestrator.predict(test_case)

        if result['success']:
            print(f"Labels: {result['predicted_labels']}")
            print(f"Confidences: {[f'{c:.2f}' for c in result['confidence_scores']]}")
            print(f"Human Review: {result['needs_human_review']}")
            print(f"Model: {result['model_used']}")
        else:
            print(f"Failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()

