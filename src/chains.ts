/**
 * CHAINS DEMO
 * -----------
 * LangChain "chains" pipe together: prompt → model → output parser.
 * This file shows three chain patterns:
 *   1. Simple LLMChain  — one prompt, one call
 *   2. Sequential chain — output of chain A feeds chain B
 *   3. Structured output chain — forces JSON with Zod schema
 */

import * as dotenv from "dotenv";
dotenv.config();

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser, JsonOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { z } from "zod";
import { getChatModel } from "./models";
import { printLangSmithInfo, getConfig } from "./config";

async function simpleLLMChain() {
  console.log("\n━━━ 1. Simple LLM Chain ━━━");

  const model = getChatModel({ temperature: 0.5 });

  // Prompt template with a variable {topic}
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a concise study assistant. Explain topics clearly in 3–5 sentences."],
    ["human", "Explain: {topic}"],
  ]);

  // Chain = prompt | model | output parser
  const chain = prompt.pipe(model).pipe(new StringOutputParser());

  const result = await chain.invoke({ topic: "What is recursion in programming?" });
  console.log("📖 Explanation:", result);
  return result;
}

async function sequentialChain() {
  console.log("\n━━━ 2. Sequential Chain ━━━");
  console.log("Chain A explains a concept → Chain B creates a quiz question from that explanation.\n");

  const model = getChatModel({ temperature: 0.7 });

  // Chain A: Explain the concept
  const explainPrompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a study assistant. Explain the concept in 2–3 sentences."],
    ["human", "Explain: {topic}"],
  ]);
  const explainChain = explainPrompt.pipe(model).pipe(new StringOutputParser());

  // Chain B: Generate a quiz question from the explanation
  const quizPrompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a quiz creator. Based on the explanation below, create one multiple-choice question with 4 options (A–D) and indicate the correct answer."],
    ["human", "Explanation:\n{explanation}\n\nCreate a quiz question:"],
  ]);
  const quizChain = quizPrompt.pipe(model).pipe(new StringOutputParser());

  // Sequential: feed explanation into quiz
  const sequentialChain = RunnableSequence.from([
    explainChain,
    (explanation: string) => quizChain.invoke({ explanation }),
  ]);

  const quizQuestion = await sequentialChain.invoke({ topic: "Binary search algorithm" });
  console.log("🧠 Quiz Question:\n", quizQuestion);
}

async function structuredOutputChain() {
  console.log("\n━━━ 3. Structured Output Chain (JSON with Zod) ━━━");

  const model = getChatModel({ temperature: 0.3 });

  // Define the expected output schema
  const StudyPlanSchema = z.object({
    topic: z.string().describe("The study topic"),
    difficulty: z.enum(["beginner", "intermediate", "advanced"]),
    estimated_hours: z.number().describe("Estimated hours to learn this topic"),
    key_concepts: z.array(z.string()).describe("List of 3–5 key concepts to cover"),
    resources: z.array(z.string()).describe("List of 2–3 recommended resources"),
  });

  // Use .withStructuredOutput() to get typed JSON back
  const structuredModel = model.withStructuredOutput(StudyPlanSchema);

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are an expert curriculum designer. Create a structured study plan."],
    ["human", "Create a study plan for: {topic}"],
  ]);

  const chain = prompt.pipe(structuredModel);

  const plan = await chain.invoke({ topic: "Machine Learning for beginners" });
  console.log("📋 Study Plan (Structured JSON):");
  console.log(JSON.stringify(plan, null, 2));
}

async function main() {
  const config = getConfig();
  printLangSmithInfo(config);

  console.log("🔗 LangChain Chains Demo\n");
  console.log("Chains solve the problem of chaining LLM calls with prompts and parsers");
  console.log("in a composable, traceable pipeline.\n");

  await simpleLLMChain();
  await sequentialChain();
  await structuredOutputChain();

  console.log("\n✅ Chains demo complete. Check LangSmith for full execution traces.");
}

main().catch(console.error);
