import os
import json
import logging
import concurrent.futures
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Environment variables
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-pro").strip()
USE_STUB = os.getenv("USE_STUB", "false").lower() in ("1", "true", "yes")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Configure Gemini API
if GOOGLE_API_KEY and not USE_STUB:
    genai.configure(api_key=GOOGLE_API_KEY)
    API_AVAILABLE = True
else:
    API_AVAILABLE = False

class Task(BaseModel):
    task_id: int
    description: str
    estimated_days: int
    depends_on_id: int = 0
    priority: Optional[str] = Field(None, description="Priority level of the task")
    skills_needed: Optional[List[str]] = Field(None, description="Required skills for the task")
    resources: Optional[List[str]] = Field(None, description="Required resources for the task")

class PlanningResponse(BaseModel):
    tasks: List[Task]
    total_duration: int
    key_milestones: Optional[List[str]] = None
    risk_factors: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None

def is_task_prompt(user_input: str) -> bool:
    """Detect if the user input is requesting a task breakdown."""
    task_keywords = [
        "plan", "task", "break down", "breakdown", "steps", "schedule",
        "organize", "create", "build", "develop", "implement", "launch",
        "how to", "what steps", "roadmap"
    ]
    return any(keyword in user_input.lower() for keyword in task_keywords)

def construct_prompt(user_input: str, is_task: bool = True) -> str:
    """Construct a dynamic prompt based on user input."""
    if is_task:
        return f"""You are an expert project manager AI assistant. Analyze this goal and create a detailed project plan:
        
GOAL: {user_input}

Provide a comprehensive breakdown including:

1. A sequence of 4-8 tasks with:
   - Clear, detailed descriptions with bullet points for specific actions
   - Realistic time estimates in days
   - Dependencies between tasks
   - Required skills or resources
   - Priority level (High/Medium/Low)

2. Additional planning insights:
   - Key milestones to watch for
   - Potential risk factors
   - Specific recommendations

Format the response as JSON:
{{
    "tasks": [
        {{
            "task_id": 1,
            "description": "Task name\\nAction Items:\\n- Specific step 1\\n- Specific step 2\\n- Specific step 3",
            "estimated_days": X,
            "depends_on_id": 0,
            "priority": "High/Medium/Low",
            "skills_needed": ["skill1", "skill2"],
            "resources": ["resource1", "resource2"]
        }}
    ],
    "total_duration": X,
    "key_milestones": ["milestone1", "milestone2"],
    "risk_factors": ["risk1", "risk2"],
    "recommendations": ["recommendation1", "recommendation2"]
}}

Make all tasks:
- Specific and actionable
- Realistically scoped
- Industry-standard and professional
- Properly sequenced with clear dependencies"""
    else:
        return f"""You are an expert consultant. Please provide detailed, professional advice for this query:

QUERY: {user_input}

Give practical, actionable guidance that is:
- Specific and implementable
- Based on industry best practices
- Backed by professional experience
- Organized in clear sections

Focus on providing real-world, concrete steps and recommendations."""

def parse_ai_response(response_text: str) -> Union[Dict, str]:
    """Parse the AI response and extract JSON if present."""
    try:
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        return response_text
    except json.JSONDecodeError:
        return response_text

def generate_task_breakdown(goal: str) -> dict:
    """Generate a comprehensive task breakdown using Gemini AI"""
    try:
        if not is_task_prompt(goal):
            return {
                "status": "ERROR", 
                "error": "This input doesn't appear to be a task-based request. Please try rephrasing with action-oriented terms."
            }

        if not API_AVAILABLE and not USE_STUB:
            return {"status": "ERROR", "error": "API is not available and stub mode is disabled"}

        if USE_STUB:
            try:
                import random

                # Task generation utilities
                def get_random_skills(count: int = 2) -> List[str]:
                    """Get random skills from predefined list."""
                    all_skills = [
                        "Research", "Planning", "Analysis", "Design", "Strategy",
                        "Project Management", "Technical Skills", "Quality Assurance",
                        "Communication", "Leadership", "Problem Solving", "Documentation",
                        "Risk Management", "Stakeholder Management", "Agile Methodologies"
                    ]
                    return random.sample(all_skills, min(count, len(all_skills)))

                def get_random_resources(count: int = 2) -> List[str]:
                    """Get random resources from predefined list."""
                    all_resources = [
                        "Project documentation", "Industry reports", "Design tools",
                        "Planning software", "Development environment", "Team collaboration platform",
                        "Testing tools", "Feedback forms", "Documentation system",
                        "Version control", "Project management software", "Communication tools"
                    ]
                    return random.sample(all_resources, min(count, len(all_resources)))

                def get_random_action_items(task_type: str, goal: str) -> List[str]:
                    """Generate relevant action items based on task type."""
                    action_items = {
                        "research": [
                            f"Research best practices for {goal}",
                            "Identify key requirements and constraints",
                            "Define project scope and boundaries",
                            "Create initial timeline and milestones",
                            "Analyze potential risks and challenges",
                            "Document research findings"
                        ],
                        "design": [
                            f"Develop detailed approach for {goal}",
                            "Create implementation plan",
                            "Define success metrics and KPIs",
                            "Design system architecture",
                            "Create technical specifications",
                            "Prepare resource allocation plan"
                        ],
                        "implementation": [
                            "Set up development environment",
                            "Implement core features and functionality",
                            "Track progress and manage issues",
                            "Coordinate with team members",
                            "Document processes and decisions",
                            "Conduct code reviews and quality checks"
                        ],
                        "testing": [
                            "Create test plans and cases",
                            "Conduct thorough testing",
                            "Gather and analyze feedback",
                            "Fix issues and make improvements",
                            "Perform performance testing",
                            "Prepare launch checklist"
                        ]
                    }
                    return random.sample(action_items.get(task_type, []), 4)

                # Generate a dynamic number of tasks
                num_tasks = random.randint(4, 6)  # Between 4-6 tasks for reasonable scope
                tasks = []
                
                # Task types and their base characteristics
                task_types = [
                    ("research", "Research & Planning", "High", 2, 8),
                    ("design", "Design & Architecture", "High", 3, 6),
                    ("implementation", "Implementation & Development", "Medium", 4, 4),
                    ("testing", "Testing & Quality Assurance", "Medium", 2, 10)
                ]

                # Add core tasks first
                for i, (task_type, name, priority, base_days, scaling) in enumerate(task_types, 1):
                    # Calculate duration based on goal complexity with some randomness
                    base_duration = max(base_days, len(goal.split()) // scaling)
                    duration = max(1, base_duration + random.randint(-1, 2))  # Add some variability
                    
                    action_items = get_random_action_items(task_type, goal)
                    
                    tasks.append({
                        "task_id": i,
                        "description": f"{name}\nAction Items:\n" + "\n".join(f"- {item}" for item in action_items),
                        "estimated_days": duration,
                        "depends_on_id": i-1 if i > 1 else 0,
                        "priority": priority,
                        "skills_needed": get_random_skills(),
                        "resources": get_random_resources()
                    })

                # Add optional tasks if num_tasks > 4
                optional_tasks = [
                    ("Documentation & Knowledge Transfer", "Low"),
                    ("Stakeholder Review & Sign-off", "Medium"),
                    ("Performance Optimization", "Medium"),
                    ("Security Review", "High")
                ]

                for i in range(len(tasks) + 1, num_tasks + 1):
                    task_name, priority = random.choice(optional_tasks)
                    duration = random.randint(2, 5)  # Random duration for optional tasks
                    
                    tasks.append({
                        "task_id": i,
                        "description": f"{task_name}\nAction Items:\n" + "\n".join([
                            f"- {item}" for item in [
                                "Create detailed documentation",
                                "Review with stakeholders",
                                "Update based on feedback",
                                "Finalize and distribute"
                            ]
                        ]),
                        "estimated_days": duration,
                        "depends_on_id": random.randint(max(1, i-2), i-1),  # Dependent on recent tasks
                        "priority": priority,
                        "skills_needed": get_random_skills(),
                        "resources": get_random_resources()
                    })

                # Calculate total duration dynamically
                total_duration = sum(task["estimated_days"] for task in tasks)

                response_data = {
                    "tasks": tasks,
                    "total_duration": total_duration,
                    "key_milestones": [
                        f"Complete research and planning for {goal}",
                        "Strategy and design approval",
                        "Core implementation finished",
                        "Final testing and launch readiness"
                    ],
                    "risk_factors": [
                        "Timeline constraints",
                        "Resource availability",
                        "Technical challenges",
                        "Stakeholder alignment"
                    ],
                    "recommendations": [
                        "Start with thorough research phase",
                        "Maintain regular stakeholder communication",
                        "Document all decisions and progress",
                        "Build in buffer time for unexpected issues"
                    ]
                }
                return {"status": "SUCCESS", "data": response_data}

            except Exception as e:
                logging.error(f"Error in stub response generation: {str(e)}")
                return {
                    "status": "ERROR",
                    "error": f"Failed to generate stub plan: {str(e)}"
                }
            return {"status": "SUCCESS", "data": response_data}

        # Use Gemini API for dynamic response
        prompt = construct_prompt(goal, is_task=True)
        model = genai.GenerativeModel('gemini-pro')
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(lambda: model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2048
                )
            ))
            try:
                response = future.result(timeout=25)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError("AI call timed out after 25 seconds")

        if not response.text:
            raise ValueError("Empty response from AI")

        # Parse the AI response
        response_data = parse_ai_response(response.text)
        if isinstance(response_data, str):
            # If we got a string instead of JSON, wrap it in a proper response
            return {
                "status": "SUCCESS",
                "data": {
                    "tasks": [{
                        "task_id": 1,
                        "description": response_data,
                        "estimated_days": 1,
                        "depends_on_id": 0
                    }]
                }
            }

        return {"status": "SUCCESS", "data": response_data}

    except Exception as e:
        logging.error(f"Error in generate_task_breakdown: {str(e)}")
        return {
            "status": "ERROR",
            "error": f"Failed to generate plan: {str(e)}"
        }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate-plan", methods=["POST"])
def generate_plan():
    try:
        data = request.get_json()
        if not data or "goal" not in data:
            return jsonify({"status": "ERROR", "error": "No goal provided"}), 400
        
        goal = data["goal"].strip()
        if not goal:
            return jsonify({"status": "ERROR", "error": "Goal cannot be empty"}), 400

        result = generate_task_breakdown(goal)
        if result.get("status") == "ERROR":
            return jsonify(result), 400

        # Validate the response data structure
        if "data" in result:
            try:
                planning_response = PlanningResponse(**result["data"])
                result["data"] = planning_response.dict()
            except Exception as e:
                logging.error(f"Response validation error: {str(e)}")
                return jsonify({
                    "status": "ERROR",
                    "error": "Invalid response format from AI"
                }), 500
            
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error in generate_plan: {str(e)}")
        return jsonify({
            "status": "ERROR",
            "error": "An unexpected error occurred"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)