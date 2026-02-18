#!/usr/bin/env python3
"""
Prompt templates and generation utilities for diverse synthetic faces.
"""

import random
from typing import Dict, List, Optional

# Demographic attributes
AGES = [
    "infant",
    "toddler", 
    "child",
    "teenager",
    "young adult",
    "middle-aged adult",
    "senior adult",
    "elderly person"
]

GENDERS = [
    "man",
    "woman",
    "person"  # Gender-neutral
]

ETHNICITIES = [
    "Caucasian",
    "African",
    "East Asian",
    "South Asian",
    "Hispanic",
    "Middle Eastern",
    "Southeast Asian",
    "Indigenous",
    "Pacific Islander",
    "mixed ethnicity"
]

# Facial features and attributes
HAIR_STYLES = [
    "short hair",
    "long hair",
    "curly hair",
    "straight hair",
    "wavy hair",
    "bald",
    "buzz cut",
    "ponytail",
    "braided hair",
    "dreadlocks"
]

FACIAL_HAIR = [
    "",  # No facial hair
    "beard",
    "mustache",
    "goatee",
    "stubble",
    "clean-shaven"
]

ACCESSORIES = [
    "",  # No accessories
    "glasses",
    "sunglasses",
    "hat",
    "cap",
    "beanie",
    "headband",
    "earrings",
    "scarf"
]

# Expressions
EXPRESSIONS = [
    "neutral expression",
    "slight smile",
    "happy expression",
    "serious expression",
    "confident expression",
    "friendly expression",
    "calm expression",
    "thoughtful expression"
]

# Lighting conditions
LIGHTING = [
    "natural lighting",
    "studio lighting",
    "soft lighting",
    "bright lighting",
    "dramatic lighting",
    "golden hour lighting",
    "overcast lighting",
    "indoor lighting"
]

# Camera angles
ANGLES = [
    "frontal view",
    "slight angle",
    "three-quarter view",
    "profile view",
    "slightly from below",
    "eye level"
]

# Image quality descriptors
QUALITY_TAGS = [
    "professional photograph",
    "high quality",
    "sharp focus",
    "detailed",
    "photorealistic",
    "8k quality",
    "professional headshot"
    "frontal view",
    "front camera",
]

def generate_random_prompt(
    include_age: bool = True,
    include_ethnicity: bool = True,
    include_accessories: bool = True,
    style: str = "professional"
) -> Dict[str, str]:
    """
    Generate a random face description prompt.
    
    Args:
        include_age: Include age descriptor
        include_ethnicity: Include ethnicity descriptor
        include_accessories: Include accessories (glasses, hats, etc.)
        style: Style of the prompt ('professional', 'casual', 'diverse')
    
    Returns:
        Dictionary with 'prompt' and 'attributes' keys
    """
    
    # Select attributes
    age = random.choice(AGES) if include_age else ""
    ethnicity = random.choice(ETHNICITIES) if include_ethnicity else ""
    gender = random.choice(GENDERS)
    
    hair = random.choice(HAIR_STYLES)
    facial_hair = random.choice(FACIAL_HAIR) if gender in ["man", "person"] else ""
    accessory = random.choice(ACCESSORIES) if include_accessories else ""
    
    expression = random.choice(EXPRESSIONS)
    lighting = random.choice(LIGHTING)
    angle = random.choice(ANGLES)
    quality = random.choice(QUALITY_TAGS)
    
    # Build prompt
    parts = ["A photorealistic portrait of"]
    
    # Demographics
    demographic_parts = []
    if age:
        demographic_parts.append(age)
    if ethnicity:
        demographic_parts.append(ethnicity)
    demographic_parts.append(gender)
    
    parts.append(" ".join(demographic_parts))
    
    # Physical attributes
    attribute_parts = []
    if hair:
        attribute_parts.append(hair)
    if facial_hair:
        attribute_parts.append(facial_hair)
    if accessory:
        attribute_parts.append(accessory)
    
    if attribute_parts:
        parts.append("with " + ", ".join(attribute_parts))
    
    # Expression and style
    parts.append(f"{expression}")
    parts.append(f"{angle}")
    parts.append(f"{lighting}")
    parts.append(f"{quality}")
    
    prompt = ", ".join(parts)
    
    # Create attributes dictionary
    attributes = {
        "age": age,
        "ethnicity": ethnicity,
        "gender": gender,
        "hair": hair,
        "facial_hair": facial_hair,
        "accessory": accessory,
        "expression": expression,
        "lighting": lighting,
        "angle": angle
    }
    
    return {
        "prompt": prompt,
        "attributes": attributes
    }

def generate_balanced_prompts(num_prompts: int) -> List[Dict[str, str]]:
    """
    Generate a balanced set of prompts ensuring demographic diversity.
    
    Args:
        num_prompts: Number of prompts to generate
    
    Returns:
        List of prompt dictionaries
    """
    prompts = []
    
    # Calculate distribution
    prompts_per_gender = num_prompts // len(GENDERS)
    prompts_per_ethnicity = num_prompts // len(ETHNICITIES)
    
    for i in range(num_prompts):
        # Cycle through genders and ethnicities for balance
        gender_idx = i % len(GENDERS)
        ethnicity_idx = (i // len(GENDERS)) % len(ETHNICITIES)
        
        # Generate prompt with forced diversity
        age = random.choice(AGES)
        gender = GENDERS[gender_idx]
        ethnicity = ETHNICITIES[ethnicity_idx]
        
        hair = random.choice(HAIR_STYLES)
        facial_hair = random.choice(FACIAL_HAIR) if gender in ["man", "person"] else ""
        accessory = random.choice(ACCESSORIES)
        
        expression = random.choice(EXPRESSIONS)
        lighting = random.choice(LIGHTING)
        angle = random.choice(ANGLES)
        quality = random.choice(QUALITY_TAGS)
        
        # Build prompt
        parts = ["A photorealistic portrait of"]
        parts.append(f"{age} {ethnicity} {gender}")
        
        attribute_parts = []
        if hair:
            attribute_parts.append(hair)
        if facial_hair:
            attribute_parts.append(facial_hair)
        if accessory:
            attribute_parts.append(accessory)
        
        if attribute_parts:
            parts.append("with " + ", ".join(attribute_parts))
        
        parts.append(f"{expression}")
        parts.append(f"{angle}")
        parts.append(f"{lighting}")
        parts.append(f"{quality}")
        
        prompt = ", ".join(parts)
        
        attributes = {
            "age": age,
            "ethnicity": ethnicity,
            "gender": gender,
            "hair": hair,
            "facial_hair": facial_hair,
            "accessory": accessory,
            "expression": expression,
            "lighting": lighting,
            "angle": angle
        }
        
        prompts.append({
            "prompt": prompt,
            "attributes": attributes
        })
    
    return prompts

def custom_prompt(
    age: Optional[str] = None,
    ethnicity: Optional[str] = None,
    gender: Optional[str] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Create a custom prompt with specific attributes.
    
    Args:
        age: Age category
        ethnicity: Ethnicity
        gender: Gender
        **kwargs: Additional attributes (hair, accessory, expression, etc.)
    
    Returns:
        Dictionary with 'prompt' and 'attributes' keys
    """
    
    age = age or random.choice(AGES)
    ethnicity = ethnicity or random.choice(ETHNICITIES)
    gender = gender or random.choice(GENDERS)
    
    hair = kwargs.get('hair', random.choice(HAIR_STYLES))
    facial_hair = kwargs.get('facial_hair', random.choice(FACIAL_HAIR) if gender == "man" else "")
    accessory = kwargs.get('accessory', random.choice(ACCESSORIES))
    expression = kwargs.get('expression', random.choice(EXPRESSIONS))
    lighting = kwargs.get('lighting', random.choice(LIGHTING))
    angle = kwargs.get('angle', random.choice(ANGLES))
    quality = kwargs.get('quality', random.choice(QUALITY_TAGS))
    
    # Build prompt
    parts = ["A photorealistic portrait of"]
    parts.append(f"{age} {ethnicity} {gender}")
    
    attribute_parts = []
    if hair:
        attribute_parts.append(hair)
    if facial_hair:
        attribute_parts.append(facial_hair)
    if accessory:
        attribute_parts.append(accessory)
    
    if attribute_parts:
        parts.append("with " + ", ".join(attribute_parts))
    
    parts.append(f"{expression}")
    parts.append(f"{angle}")
    parts.append(f"{lighting}")
    parts.append(f"{quality}")
    
    prompt = ", ".join(parts)
    
    attributes = {
        "age": age,
        "ethnicity": ethnicity,
        "gender": gender,
        "hair": hair,
        "facial_hair": facial_hair,
        "accessory": accessory,
        "expression": expression,
        "lighting": lighting,
        "angle": angle
    }
    
    return {
        "prompt": prompt,
        "attributes": attributes
    }

# Example usage
if __name__ == "__main__":
    # Test random prompt generation
    print("Random Prompt Examples:")
    print("=" * 80)
    for i in range(5):
        result = generate_random_prompt()
        print(f"\n{i+1}. {result['prompt']}")
        print(f"   Attributes: {result['attributes']}")
    
    print("\n\n" + "=" * 80)
    print("Balanced Prompt Set (10 prompts):")
    print("=" * 80)
    balanced = generate_balanced_prompts(10)
    for i, result in enumerate(balanced):
        print(f"\n{i+1}. {result['prompt'][:100]}...")
