CHEATSHEET = """# GENERATOR (PROBLEM SOLVER)

Instruction: You are an expert problem-solving assistant tasked with analyzing and solving various questions using a combination of your expertise and provided reference materials. Each task will include:
1. A specific question or problem to solve
2. A cheatsheet containing relevant strategies, patterns, and examples from similar problems

---

## 1. ANALYSIS & STRATEGY

- Carefully analyze both the question and cheatsheet before starting
- Search for and identify any applicable patterns, strategies, or examples within the cheatsheet
- Create a structured approach to solving the problem at hand
- Review and document any limitations in the provided reference materials

## 2. SOLUTION DEVELOPMENT

- Present your solution using clear, logical steps that others can follow and review
- Explain your reasoning and methodology before presenting final conclusions
- Provide detailed explanations for each step of the process
- Check and verify all assumptions and intermediate calculations

## 3. FINAL ANSWER FORMAT

ALWAYS present your final answer in the following format:

\\boxed{<answer>}

Example:
Q: What is the meaning of life?
A: [...] My final answer is \\boxed{42}.

-----

CHEATSHEET:
"""

CHEATSHEET_SUFFIX = """

-----
-----

Now it is time to solve the following question.

CURRENT INPUT:
"""

def build_conv(env, query, hint=None):
    if env == "R1":
        dialogue = [
            {
                "role": "user",
                "content": query,
            }
        ]
    elif env == "hint-generator":
        dialogue = [
            {
                "role": "user",
                "content": f"Given the problem, generate a concise hint providing the most valuable insight for solving this problem.\n{query}"
            }
        ]
    elif env == "hint-generator-v1":
        dialogue = [
            {
                "role": "user",
                "content": f"""Given the following math problem, generate a list of insightful hints that help guide a student toward solving the problem. Each hint should be wrapped in a <note> block with the following structure:

<note>
<description>[Brief explanation of a key idea or technique relevant to the problem]</description>
<example>[Concrete illustrative example that demonstrates the idea in action]</example>
</note>
Combine all hint blocks inside a <notes> element. Your goal is to help the student reason through the problem step-by-step by surfacing useful strategies, intermediate goals, or simplifications.

Problem: {query}"""
            }
        ]
    elif env == "cheatsheet":
        
        if hint == None:
            prompt = query
        else:
            prompt = CHEATSHEET
            prompt += hint
            prompt += CHEATSHEET_SUFFIX
            prompt += query
                
        dialogue = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    return dialogue
