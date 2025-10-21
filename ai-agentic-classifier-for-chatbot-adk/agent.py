import os
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.models import Gemini
from typing import List, Dict, Any
import logging
import asyncio

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleADKClassifier:
    """Simple working ADK classifier without problematic imports."""

    def __init__(self, label_schema: Dict[str, Any]):
        self.labels = label_schema['labels']

        # Initialize the model
        self.model = Gemini(model="gemini-2.0-flash")

        # Create the agent
        self.agent = Agent(
            name="ChatbotClassifier",
            model=self.model,
            instruction=self._build_instruction(),
            output_format={
                "type": "object",
                "properties": {
                    "predicted_labels": {"type": "array", "items": {"type": "string"}},
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number"},
                    "risk_assessment": {"type": "string"}
                },
                "required": ["predicted_labels", "reasoning", "confidence"]
            }
        )

    def _build_instruction(self) -> str:
        return f"""
        You are a multi-label chatbot message classifier. Analyze the user message and assign ALL relevant labels.

        AVAILABLE LABELS: {", ".join(self.labels)}

        CLASSIFICATION RULES:
        1. Be precise - only assign labels that clearly apply
        2. Consider both explicit and implicit meanings
        3. Assign multiple labels when appropriate (multi-label classification)
        4. Be conservative - when in doubt, don't assign
        5. Consider user intent and emotional tone

        RISK ASSESSMENT:
        - Mark as high risk if: complaint, urgent, emergency, or low confidence
        - Mark as medium risk if: billing, technical issues, refund requests  
        - Mark as low risk if: greeting, question, feedback

        Return a JSON object with:
        - predicted_labels: array of applicable labels
        - reasoning: brief explanation for your choices
        - confidence: overall confidence score (0.0-1.0)
        - risk_assessment: low/medium/high (optional)

        Example:
        Input: "Hello, I need urgent help with my billing statement from last month"
        Output: {{
          "predicted_labels": ["greeting", "billing_help", "urgent"],
          "reasoning": "Message contains greeting, billing reference, and urgent language",
          "confidence": 0.88,
          "risk_assessment": "high"
        }}
        """

    async def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Run classification using ADK agent."""
        try:
            message = request['message']
            context = request.get('context', {})

            logger.info(f"Classifying: {message[:100]}...")

            # Add context if available
            full_input = message
            if context.get('conversation_history'):
                full_input = f"Conversation history: {context['conversation_history']}\n\nCurrent message: {message}"

            # Run the agent
            result = await self.agent.run(input_content=full_input)

            # Process the result
            return self._process_result(result)

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                'predicted_labels': [],
                'confidence_scores': [],
                'needs_human_review': True,
                'error': str(e),
                'success': False
            }

    def _process_result(self, result: Any) -> Dict[str, Any]:
        """Process ADK agent result."""
        try:
            # Extract data from result - handle different ADK response structures
            if hasattr(result, 'output'):
                data = result.output
            else:
                data = result

            # Handle nested data structure
            if hasattr(data, 'data'):
                result_data = data.data
            else:
                result_data = data

            # Extract fields with defaults
            predicted_labels = result_data.get('predicted_labels', [])
            confidence = result_data.get('confidence', 0.5)
            reasoning = result_data.get('reasoning', 'No reasoning provided')
            risk_assessment = result_data.get('risk_assessment', 'medium')

            # Validate labels
            valid_labels = [label for label in predicted_labels if label in self.labels]

            # Create confidence scores
            confidence_scores = [confidence] * len(valid_labels)

            # Determine if human review is needed
            needs_human_review = self._should_escalate(valid_labels, confidence, risk_assessment)

            return {
                'predicted_labels': valid_labels,
                'confidence_scores': confidence_scores,
                'needs_human_review': needs_human_review,
                'reasoning': reasoning,
                'risk_assessment': risk_assessment,
                'framework': 'google-adk',
                'success': True
            }

        except Exception as e:
            logger.error(f"Result processing failed: {e}")
            return {
                'predicted_labels': [],
                'confidence_scores': [],
                'needs_human_review': True,
                'error': f"Processing failed: {str(e)}",
                'success': False
            }

    def _should_escalate(self, labels: List[str], confidence: float, risk: str) -> bool:
        """Determine if human review is needed."""
        high_risk_labels = {'complaint', 'urgent', 'emergency', 'refund_request'}

        if (confidence < 0.6 or
                risk == 'high' or
                any(label in high_risk_labels for label in labels) or
                len(labels) == 0):
            return True
        return False


# Multi-agent version without SequentialAgent if it's problematic
class MultiAgentADKClassifier:
    """Multi-agent classifier that runs agents sequentially manually."""

    def __init__(self, label_schema: Dict[str, Any]):
        self.labels = label_schema['labels']
        self.model = GeminiModel(model="gemini-2.0-flash")

        # Create individual agents
        self.classifier = Agent(
            name="Classifier",
            model=self.model,
            instruction=f"Classify this message using labels: {', '.join(self.labels)}",
            output_format={"type": "object", "properties": {
                "labels": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"}
            }}
        )

        self.confidence_assessor = Agent(
            name="ConfidenceAssessor",
            model=self.model,
            instruction="Assess confidence for each label (0.0-1.0)",
            output_format={"type": "object", "properties": {
                "confidences": {"type": "object", "additionalProperties": {"type": "number"}},
                "overall_confidence": {"type": "number"}
            }}
        )

    async def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-agent classification manually."""
        try:
            message = request['message']

            # Step 1: Classification
            classification_result = await self.classifier.run(input_content=message)
            labels = classification_result.output.get('labels', [])
            reasoning = classification_result.output.get('reasoning', '')

            # Step 2: Confidence assessment
            confidence_input = f"Labels: {labels}\nReasoning: {reasoning}\nMessage: {message}"
            confidence_result = await self.confidence_assessor.run(input_content=confidence_input)
            confidences = confidence_result.output.get('confidences', {})
            overall_confidence = confidence_result.output.get('overall_confidence', 0.5)

            # Validate and process
            valid_labels = [label for label in labels if label in self.labels]
            confidence_scores = [confidences.get(label, overall_confidence) for label in valid_labels]

            needs_review = overall_confidence < 0.7 or len(valid_labels) == 0

            return {
                'predicted_labels': valid_labels,
                'confidence_scores': confidence_scores,
                'needs_human_review': needs_review,
                'reasoning': reasoning,
                'success': True
            }

        except Exception as e:
            logger.error(f"Multi-agent classification failed: {e}")
            return {
                'predicted_labels': [],
                'confidence_scores': [],
                'needs_human_review': True,
                'error': str(e),
                'success': False
            }


# Test function
async def main():
    """Test the ADK classifier."""

    # Make sure API key is set
    if not os.getenv('GOOGLE_API_KEY'):
        print("Please set GOOGLE_API_KEY environment variable")
        return

    label_schema = {
        'labels': [
            'greeting', 'question', 'complaint', 'billing_help',
            'technical_support', 'feedback', 'refund_request', 'urgent'
        ]
    }

    print("Testing ADK Classifier")
    print("=" * 50)

    # Test with simple classifier
    classifier = SimpleADKClassifier(label_schema)

    test_cases = [
        {'message': "Hello, I need help with my billing statement from last month", 'context': {}},
        {'message': "This app is completely broken and I want my money back now!", 'context': {}},
        {'message': "How do the premium features work?", 'context': {}},
        {'message': "Thanks for your help yesterday!", 'context': {}}
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{test_case['message']}'")
        print("-" * 40)

        result = await classifier.predict(test_case)

        if result['success']:
            print(f"Labels: {result['predicted_labels']}")
            print(f"Confidences: {[f'{c:.2f}' for c in result['confidence_scores']]}")
            print(f"Human Review: {result['needs_human_review']}")
            print(f"Risk: {result.get('risk_assessment', 'N/A')}")
            print(f"Reasoning: {result['reasoning'][:80]}...")
        else:
            print(f"Failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
