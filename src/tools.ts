/**
 * CUSTOM TOOLS FOR THE AGENT
 * --------------------------
 * Tools are functions the agent can call to take actions or fetch data.
 * Each tool has: a name, a description (read by the LLM), and an execute function.
 */

import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Tool 1: Calculator — handles math without hallucination
export const calculatorTool = tool(
  async ({ expression }: { expression: string }) => {
    try {
      // Safe evaluation using Function constructor (avoid eval())
      const result = Function(`"use strict"; return (${expression})`)();
      return `The result of ${expression} = ${result}`;
    } catch {
      return `Error: Could not evaluate "${expression}". Please use valid math expressions.`;
    }
  },
  {
    name: "calculator",
    description: "Evaluates mathematical expressions. Use for any calculations. Input: a math expression like '2 + 2' or '(15 * 8) / 3'.",
    schema: z.object({
      expression: z.string().describe("A mathematical expression to evaluate"),
    }),
  }
);

// Tool 2: Topic explainer — returns a structured explanation
export const topicExplainerTool = tool(
  async ({ topic, level }: { topic: string; level: string }) => {
    // This simulates a knowledge base lookup (in real apps: vector DB retrieval)
    const knowledgeBase: Record<string, Record<string, string>> = {
      "binary search": {
        beginner: "Binary search finds an item in a sorted list by repeatedly halving the search range. It's much faster than checking every item.",
        intermediate: "Binary search is an O(log n) algorithm that works on sorted arrays. It compares the target to the midpoint and eliminates half the remaining elements each iteration.",
        advanced: "Binary search has O(log n) time complexity. Key variants: lower_bound, upper_bound for duplicate handling. Common pitfalls: integer overflow in mid = (lo + hi) / 2 (use lo + (hi - lo) / 2 instead).",
      },
      "recursion": {
        beginner: "Recursion is when a function calls itself. It needs a base case to stop, otherwise it runs forever.",
        intermediate: "Recursive functions break problems into smaller sub-problems. Each call adds a stack frame — deep recursion can cause stack overflow.",
        advanced: "Tail recursion can be optimized by compilers. Memoization converts recursive solutions from exponential to polynomial time (e.g., Fibonacci: O(2^n) → O(n)).",
      },
    };

    const topicLower = topic.toLowerCase();
    const levelLower = level.toLowerCase() as "beginner" | "intermediate" | "advanced";

    for (const key of Object.keys(knowledgeBase)) {
      if (topicLower.includes(key)) {
        return knowledgeBase[key][levelLower] || knowledgeBase[key]["intermediate"];
      }
    }
    return `I don't have a pre-built explanation for "${topic}" at ${level} level. The AI will generate one.`;
  },
  {
    name: "topic_explainer",
    description: "Retrieves explanations for common CS topics at different difficulty levels. Use this before generating your own explanation. Topics include: binary search, recursion, sorting, linked lists, trees, graphs.",
    schema: z.object({
      topic: z.string().describe("The programming or CS topic to explain"),
      level: z.enum(["beginner", "intermediate", "advanced"]).describe("The explanation difficulty level"),
    }),
  }
);

// Tool 3: Study tip generator
export const studyTipTool = tool(
  async ({ subject }: { subject: string }) => {
    const tips: Record<string, string[]> = {
      "math": [
        "Practice problems daily — math is a skill, not memorization.",
        "Show all steps; partial credit often comes from method even when the answer is wrong.",
        "Use spaced repetition for formulas.",
      ],
      "programming": [
        "Type out code examples — don't just read them.",
        "Build a small project with each new concept.",
        "Use rubber duck debugging: explain your code to an imaginary listener.",
      ],
      "default": [
        "Use active recall: test yourself instead of re-reading notes.",
        "Study in 25-minute focused sessions (Pomodoro technique).",
        "Teach the concept to someone else — it reveals gaps in your knowledge.",
      ],
    };

    const subjectLower = subject.toLowerCase();
    const matched = Object.keys(tips).find((k) => subjectLower.includes(k));
    const selectedTips = tips[matched || "default"];

    return `Study tips for ${subject}:\n` + selectedTips.map((t, i) => `${i + 1}. ${t}`).join("\n");
  },
  {
    name: "study_tips",
    description: "Returns effective study tips tailored to a specific subject (e.g., math, programming, biology).",
    schema: z.object({
      subject: z.string().describe("The subject the student is studying"),
    }),
  }
);

export const allTools = [calculatorTool, topicExplainerTool, studyTipTool];
