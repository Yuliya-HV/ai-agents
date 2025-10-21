"""
Agent-Based Classification with Structured JSON Output - LLM OpenAI
"""


import openai
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential

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

class StructuredLLMAgent:
    """Base agent with structured JSON output."""

    def __init__(self, role: AgentRole, model_config: Dict[str, Any]):
        self.role = role
        self.model_name = model_config.get('model_name', 'gpt-3.5-turbo')
        self.temperature = model_config.get('temperature', 0.1)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def _call_llm_structured(self, system_prompt: str, user_message: str, response_format: Dict) -> Dict[str, Any]:
        """Call LLM with forced JSON response format."""
        try:
            # Use response_format to force JSON output
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},  # Force JSON output
                max_tokens=1000
            )

            response_text = response.choices[0].message.content.strip()
            return json.loads(response_text)

        except json.JSONDecodeError as e:
            logger.error(f"Agent {self.role.value} returned invalid JSON: {response_text}")
            raise
        except Exception as e:
            logger.error(f"Agent {self.role.value} failed: {e}")
            raise

class ClassifierAgent(StructuredLLMAgent):
    """Primary agent for label classification with structured output."""

    def classify(self, message: str, labels: List[str], context: Dict[str, Any] = None) -> AgentResponse:
        """Perform multi-label classification with structured JSON output."""

        system_prompt = self._build_classification_system_prompt(labels, context)
        user_message = f"Message to classify: {message}"

        response_format = {
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
                    "description": "Overall confidence in classification (0-1)"
                }
            },
            "required": ["assigned_labels", "reasoning", "confidence"]
        }

        try:
            result = self._call_llm_structured(system_prompt, user_message, response_format)

            # Validate labels are in our schema
            valid_labels = [label for label in result['assigned_labels'] if label in labels]

            return AgentResponse(
                role=self.role,
                content=valid_labels,
                confidence=result['confidence'],
                reasoning=result['reasoning']
            )

        except Exception as e:
            logger.error(f"Classifier agent failed: {e}")
            # Fallback response
            return AgentResponse(
                role=self.role,
                content=[],
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}"
            )

    def _build_classification_system_prompt(self, labels: List[str], context: Dict[str, Any]) -> str:
        """Build system prompt for classification."""

        context_str = ""
        if context and context.get('conversation_history'):
            context_str = f"\nConversation Context: {context['conversation_history']}"

        return f"""
        You are an expert message classifier. Analyze the user message and identify ALL applicable labels.

        AVAILABLE LABELS: {json.dumps(labels)}

        {context_str}

        CLASSIFICATION GUIDELINES:
        1. Be precise - only assign labels that clearly apply
        2. Consider both explicit and implicit meanings  
        3. Assign multiple labels when appropriate
        4. Consider conversation context and user intent
        5. Be conservative - when in doubt, don't assign

        OUTPUT FORMAT: JSON with:
        - assigned_labels: array of applicable labels
        - reasoning: brief explanation for each assigned label
        - confidence: overall confidence score (0.0-1.0)

        Return ONLY valid JSON. Do not include any other text.
        """

class ValidatorAgent(StructuredLLMAgent):
    """Agent to validate and refine classifications with structured output."""

    def validate(self, message: str, proposed_labels: List[str], classifier_reasoning: str, labels: List[str]) -> AgentResponse:
        """Validate proposed labels with structured JSON output."""

        system_prompt = self._build_validation_system_prompt(labels)
        user_message = f"""
        Message: {message}
        Proposed Labels: {proposed_labels}
        Classifier Reasoning: {classifier_reasoning}
        """

        response_format = {
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
                    "description": "Confidence in validation (0-1)"
                },
                "changes_made": {
                    "type": "object",
                    "properties": {
                        "added": {"type": "array", "items": {"type": "string"}},
                        "removed": {"type": "array", "items": {"type": "string"}},
                        "kept": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["validated_labels", "validation_notes", "validation_confidence"]
        }

        try:
            result = self._call_llm_structured(system_prompt, user_message, response_format)

            # Validate labels
            valid_labels = [label for label in result['validated_labels'] if label in labels]

            return AgentResponse(
                role=self.role,
                content=valid_labels,
                confidence=result['validation_confidence'],
                reasoning=result['validation_notes']
            )

        except Exception as e:
            logger.error(f"Validator agent failed: {e}")
            # If validation fails, return original labels with low confidence
            return AgentResponse(
                role=self.role,
                content=proposed_labels,
                confidence=0.3,
                reasoning=f"Validation failed, using original labels: {str(e)}"
            )

    def _build_validation_system_prompt(self, labels: List[str]) -> str:
        return f"""
        You are a classification validator. Review the proposed labels and reasoning.

        AVAILABLE LABELS: {json.dumps(labels)}

        Your Task:
        1. Verify each proposed label is appropriate
        2. Suggest any missing labels from the available list
        3. Remove any incorrect labels
        4. Provide validation notes explaining changes

        OUTPUT FORMAT: JSON with:
        - validated_labels: final list of labels
        - validation_notes: your assessment and changes
        - validation_confidence: your confidence in this validation (0.0-1.0)
        - changes_made: object showing added/removed/kept labels (optional)

        Return ONLY valid JSON. Do not include any other text.
        """

class ConfidenceAssessorAgent(StructuredLLMAgent):
    """Agent to assess confidence scores with structured output."""

    def assess_confidence(self, message: str, final_labels: List[str],
                          classifier_reasoning: str, validator_notes: str) -> AgentResponse:
        """Assess individual confidence scores with structured JSON."""

        system_prompt = self._build_confidence_system_prompt()
        user_message = f"""
        Message: {message}
        Final Labels: {final_labels}
        Classifier Reasoning: {classifier_reasoning}
        Validator Notes: {validator_notes}
        """

        response_format = {
            "type": "object",
            "properties": {
                "confidence_scores": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Confidence score for each label (0.0-1.0)"
                },
                "assessment_notes": {
                    "type": "string",
                    "description": "Brief explanation of confidence assessments"
                },
                "overall_confidence": {
                    "type": "number",
                    "description": "Overall confidence in these scores (0.0-1.0)"
                }
            },
            "required": ["confidence_scores", "assessment_notes", "overall_confidence"]
        }

        try:
            result = self._call_llm_structured(system_prompt, user_message, response_format)

            # Validate and clamp confidence scores
            validated_scores = {}
            for label in final_labels:
                score = result['confidence_scores'].get(label, 0.5)
                validated_scores[label] = max(0.0, min(1.0, score))

            return AgentResponse(
                role=self.role,
                content=validated_scores,
                confidence=result['overall_confidence'],
                reasoning=result['assessment_notes']
            )

        except Exception as e:
            logger.error(f"Confidence assessor failed: {e}")
            # Fallback: equal medium confidence
            fallback_scores = {label: 0.7 for label in final_labels}
            return AgentResponse(
                role=self.role,
                content=fallback_scores,
                confidence=0.5,
                reasoning=f"Confidence assessment failed, using fallback: {str(e)}"
            )

    def _build_confidence_system_prompt(self) -> str:
        return """
        You are a confidence assessor. Assign individual confidence scores for each label.

        Considerations:
        - Explicit vs implicit evidence in the message
        - Strength of supporting reasoning from classifier/validator
        - Ambiguity or multiple interpretations possible
        - Consistency with conversation context

        Scoring Guidelines:
        - 0.9-1.0: Very strong, unambiguous evidence
        - 0.7-0.8: Strong evidence, minor ambiguity
        - 0.5-0.6: Moderate evidence, some ambiguity
        - 0.3-0.4: Weak evidence, significant ambiguity  
        - 0.0-0.2: Very weak evidence, high ambiguity

        OUTPUT FORMAT: JSON with:
        - confidence_scores: object mapping each label to score (0.0-1.0)
        - assessment_notes: brief explanation of scores
        - overall_confidence: your confidence in these assessments (0.0-1.0)

        Return ONLY valid JSON. Do not include any other text.
        """

class EscalationManagerAgent(StructuredLLMAgent):
    """Agent to decide human review with structured output."""

    def assess_escalation(self, message: str, labels: List[str],
                          confidence_scores: Dict[str, float], all_reasoning: Dict[str, str]) -> AgentResponse:
        """Determine if human review is needed with structured JSON."""

        system_prompt = self._build_escalation_system_prompt()
        user_message = f"""
        Message: {message}
        Assigned Labels: {labels}
        Confidence Scores: {json.dumps(confidence_scores)}
        Classifier Reasoning: {all_reasoning.get('classifier', 'None')}
        Validator Reasoning: {all_reasoning.get('validator', 'None')}
        """

        response_format = {
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
                },
                "triggering_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Factors that influenced the decision"
                }
            },
            "required": ["needs_human_review", "escalation_reason", "escalation_confidence"]
        }

        try:
            result = self._call_llm_structured(system_prompt, user_message, response_format)

            return AgentResponse(
                role=self.role,
                content=result['needs_human_review'],
                confidence=result['escalation_confidence'],
                reasoning=result['escalation_reason']
            )

        except Exception as e:
            logger.error(f"Escalation manager failed: {e}")
            # Conservative fallback: escalate on failure
            return AgentResponse(
                role=self.role,
                content=True,
                confidence=0.5,
                reasoning=f"Escalation assessment failed, defaulting to human review: {str(e)}"
            )

    def _build_escalation_system_prompt(self) -> str:
        return """
        You are an escalation manager. Decide if this classification needs human review.

        Escalation Criteria (escalate if ANY are true):
        - Any confidence score < 0.6
        - High-risk labels present (complaint, legal, emergency, security)
        - Contradictory reasoning between classifier and validator
        - Message contains urgent language (immediately, emergency, ASAP)
        - Low overall confidence in the classification

        OUTPUT FORMAT: JSON with:
        - needs_human_review: boolean (true/false)
        - escalation_reason: explanation for decision
        - escalation_confidence: your confidence in this decision (0.0-1.0)
        - triggering_factors: array of factors that influenced decision (optional)

        Return ONLY valid JSON. Do not include any other text.
        """

class StructuredAgentOrchestrator:
    """Orchestrator with guaranteed JSON responses."""

    def __init__(self, model_config: Dict[str, Any], label_schema: Dict[str, Any]):
        self.model_config = model_config
        self.labels = label_schema['labels']

        # Initialize agents with structured output
        self.agents = {
            AgentRole.CLASSIFIER: ClassifierAgent(AgentRole.CLASSIFIER, model_config),
            AgentRole.VALIDATOR: ValidatorAgent(AgentRole.VALIDATOR, model_config),
            AgentRole.CONFIDENCE_ASSESSOR: ConfidenceAssessorAgent(AgentRole.CONFIDENCE_ASSESSOR, model_config),
            AgentRole.ESCALATION_MANAGER: EscalationManagerAgent(AgentRole.ESCALATION_MANAGER, model_config),
        }

        logger.info(f"Initialized structured agent orchestrator for {len(self.labels)} labels")

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-agent classification with structured JSON responses."""
        try:
            message = request['message']
            context = request.get('context', {})

            logger.info(f"Starting structured agent classification: {message[:100]}...")

            # Step 1: Initial Classification
            classifier_response = self.agents[AgentRole.CLASSIFIER].classify(
                message, self.labels, context
            )
            logger.info \
                (f"Classifier: {len(classifier_response.content)} labels, confidence {classifier_response.confidence:.2f}")

            # Step 2: Validation
            validator_response = self.agents[AgentRole.VALIDATOR].validate(
                message, classifier_response.content, classifier_response.reasoning, self.labels
            )
            logger.info \
                (f"Validator: {len(validator_response.content)} labels, confidence {validator_response.confidence:.2f}")

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
                'processing_steps': 4,
                'success': True
            }

        except Exception as e:
            logger.error(f"Structured agent pipeline failed: {e}")
            return {
                'predicted_labels': [],
                'confidence_scores': [],
                'needs_human_review': True,
                'error': str(e),
                'processing_steps': 0,
                'success': False
            }




# ------------------------- HOW TO USE THE CODE -------------------------

"""STRUCTURED AGENT CLASSIFICATION (JSON OUTPUT) APPROACH

    WHY THIS IS USEFUL?
    
    KEY FEATURES:
    - Guaranteed JSON responses from all agents
    - Structured output validation
    - Robust error handling with fallbacks
    - Consistent confidence scoring
    - Reliable escalation decisions
    
"""

def demo_structured_agents():

    # Test config
    model_config = {
        'model_name': 'gpt-3.5-turbo',
        'temperature': 0.1,
        'api_key': 'YOUR_OPENAI_KEY'  # add your key here
    }

    label_schema = {
        'labels': [
            'greeting', 'question', 'complaint', 'billing_help',
            'technical_support', 'feedback', 'refund_request', 'urgent'
        ]
    }

    # Initialize orchestrator
    orchestrator = StructuredAgentOrchestrator(model_config, label_schema)

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
            'message': "Can you explain how the premium features work?",
            'context': {}
        }
    ]

    print("RUNNING STRUCTURED AGENT PIPELINE:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST CASE {i}: '{test_case['message']}'")
        print(f"{'=' * 70}")

        result = orchestrator.predict(test_case)

        if result['success']:
            print(f"FINAL LABELS: {result['predicted_labels']}")
            print(f"CONFIDENCES: {[f'{c:.2f}' for c in result['confidence_scores']]}")
            print(f"HUMAN REVIEW: {result['needs_human_review']}")
            print(f"PROCESSING STEPS: {result['processing_steps']}")

            print("\nAGENT CONFIDENCES:")
            for agent, confidence in result['agent_confidences'].items():
                print(f"   {agent.upper()}: {confidence:.2f}")

            print("\nAGENT REASONING SUMMARY:")
            for agent, reasoning in result['agent_reasoning'].items():
                print(f"   {agent.upper()}: {reasoning[:80]}...")
        else:
            print(f"PIPELINE FAILED: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    demo_structured_agents()

