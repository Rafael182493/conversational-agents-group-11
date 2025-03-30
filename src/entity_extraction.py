import torch
import json
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


class EntityExtractor:
    """Extract entities from text using a large language model."""


    # NOTE if we go & use cloudllm anyway we can just swap that in here as well ig
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """Initialize the entity extractor with a language model."""
        print(f"Loading entity extraction model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        #currently skippin' this for testing purposes.
        #you can remove the return
        return

        try:
            print("flot16 using we are")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        except (ImportError, RuntimeError):
            print("flot32 using we are because error")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )


    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract event planning entities from text.
        Returns a dictionary of entity types and their values.
        """
        return {}
        if not text or text.strip() == "":
            return {}

        try:
            # 2-shot prompt with examples
            prompt = f"""<|im_start|>user
Extract event planning entities from texts. Format as JSON.

Example 1:
Text: "We'll have our team meeting on April 5th at 3PM in Conference Room B. John Smith will present the Q1 results. Budget is $500 for refreshments from Good Eats Catering."

Output:
{{"date": "April 5th", "time": "3PM", "location": "Conference Room B", "people": ["John Smith"], "organizations": ["Good Eats Catering"], "budget": "$500", "purpose": "team meeting and Q1 results presentation"}}

Example 2:
Text: "Reminder: The charity gala is scheduled for Saturday, December 10th, 7-10PM at the Hilton Downtown. Contact Sarah Jones (sjones@email.com) for tickets ($150 each). Proceeds benefit City Food Bank."

Output:
{{"date": "December 10th", "day": "Saturday", "time": "7-10PM", "location": "Hilton Downtown", "people": ["Sarah Jones"], "contact": "sjones@email.com", "ticket_price": "$150", "beneficiary": "City Food Bank", "event_type": "charity gala"}}

Now extract entities from this text:
{text}
<|im_end|>
<|im_start|>assistant
"""

            # Generate response
            input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                input_ids.input_ids,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            response = self.tokenizer.decode(output[0], skip_special_tokens=False)

            # Parse the response to get JSON content
            try:
                assistant_response = response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
                entities = json.loads(assistant_response)
                return entities
            except (IndexError, json.JSONDecodeError):
                # Try to extract JSON using regex if normal parsing fails
                import re
                json_match = re.search(r'({.*})', response, re.DOTALL)
                if json_match:
                    try:
                        entities = json.loads(json_match.group(1))
                        return entities
                    except json.JSONDecodeError:
                        pass

                print("Could not parse entity extraction response")
                return {}

        except Exception as e:
            print(f"Error in entity extraction: {str(e)}")
            return {}