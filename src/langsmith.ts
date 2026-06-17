/**
 * LANGSMITH TRACING DEMO
 * ----------------------
 * LangSmith is LangChain's observability platform. It records every:
 *   - LLM call (prompt in, completion out, tokens used, latency)
 *   - Chain execution (which steps ran, inputs/outputs at each step)
 *   - Tool invocation (which tool, what args, what result)
 *   - Error (where it happened and why)
 *
 * This file shows how to use traceable() to wrap custom functions,
 * giving you full visibility even for non-LangChain code.
 */

import * as dotenv from "dotenv";
dotenv.config();

import { traceable } from "langsmith/traceable";
import { wrapOpenAI } from "langsmith/wrappers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { getChatModel } from "./models";
import { printLangSmithInfo, getConfig } from "./config";

// Example 1: Wrap a custom function with @traceable
// This appears as a named span in LangSmith traces
const generateStudyPlan = traceable(
  async (topic: string, durationWeeks: number): Promise<string> => {
    const model = getChatModel({ temperature: 0.5 });

    const prompt = ChatPromptTemplate.fromMessages([
      ["system", "You are a curriculum designer. Create a concise study plan."],
      ["human", "Create a {weeks}-week study plan for: {topic}. List week-by-week goals only."],
    ]);

    const chain = prompt.pipe(model).pipe(new StringOutputParser());
    return chain.invoke({ topic, weeks: durationWeeks.toString() });
  },
  { name: "generate_study_plan", run_type: "chain" }
);

// Example 2: Nested traceable calls — LangSmith shows parent/child spans
const evaluateAnswer = traceable(
  async (question: string, studentAnswer: string): Promise<{ score: number; feedback: string }> => {
    const model = getChatModel({ temperature: 0.2 });

    const prompt = ChatPromptTemplate.fromMessages([
      ["system", "You are a strict but fair grader. Evaluate the student's answer."],
      ["human", `Question: {question}\n\nStudent's answer: {answer}\n\nRespond with JSON only: {{"score": 0-10, "feedback": "brief feedback"}}`],
    ]);

    const chain = prompt.pipe(model).pipe(new StringOutputParser());
    const raw = await chain.invoke({ question, answer: studentAnswer });

    // Extract JSON from the response
    const match = raw.match(/\{[\s\S]*\}/);
    if (match) {
      return JSON.parse(match[0]);
    }
    return { score: 5, feedback: "Unable to parse evaluation." };
  },
  { name: "evaluate_student_answer", run_type: "chain" }
);

// Example 3: Parent trace that calls child traces — see full tree in LangSmith
const runHomeworkSession = traceable(
  async (studentName: string) => {
    console.log(`\n📚 Starting homework session for ${studentName}...\n`);

    // Child trace 1: Generate study plan
    console.log("Step 1: Generating study plan...");
    const plan = await generateStudyPlan("TypeScript and Node.js", 4);
    console.log("📋 Study Plan:\n", plan);

    // Child trace 2: Evaluate a sample answer
    console.log("\nStep 2: Evaluating sample answer...");
    const evaluation = await evaluateAnswer(
      "What is a closure in JavaScript?",
      "A closure is when a function remembers variables from its outer scope even after the outer function has returned."
    );
    console.log(`📝 Evaluation — Score: ${evaluation.score}/10`);
    console.log(`   Feedback: ${evaluation.feedback}`);

    return { plan, evaluation, student: studentName };
  },
  { name: "homework_session", run_type: "chain" }
);

// Example 4: Error tracing — LangSmith captures failures too
const demonstrateErrorTracing = traceable(
  async () => {
    console.log("\n━━━ Error Tracing Demo ━━━");
    console.log("LangSmith captures errors with full context for debugging.\n");

    const model = getChatModel({ temperature: 0 });

    // Intentionally malformed prompt to show error handling
    try {
      const prompt = ChatPromptTemplate.fromMessages([
        ["system", "You are a helpful assistant."],
        ["human", "This is a normal request: {valid_variable}"],
      ]);

      // Passing wrong variable name — will cause a template error
      const chain = prompt.pipe(model).pipe(new StringOutputParser());
      await chain.invoke({ wrong_variable: "test" }); // wrong key!
    } catch (error) {
      const err = error as Error;
      console.log("❌ Caught error (visible in LangSmith trace):", err.message);
      console.log("🔍 In LangSmith, you'd see exactly which step failed and why.");
    }
  },
  { name: "demonstrate_error", run_type: "chain" }
);

async function main() {
  const config = getConfig();
  printLangSmithInfo(config);

  if (!config.langsmithEnabled) {
    console.log("⚠️  LangSmith is not configured. Traces won't appear in the dashboard.");
    console.log("   Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in your .env file.\n");
  }

  console.log("🔍 LangSmith Tracing Demo\n");
  console.log("All function calls below create traces visible at https://smith.langchain.com\n");

  // Run the homework session (creates nested parent/child traces)
  await runHomeworkSession("Alice");

  // Demonstrate error tracing
  await demonstrateErrorTracing();

  console.log("\n✅ Tracing demo complete!");
  if (config.langsmithEnabled) {
    console.log(`\n📊 View your traces at: https://smith.langchain.com`);
    console.log(`   Project: ${config.langsmithProject}`);
    console.log("\nWhat to look for in LangSmith:");
    console.log("  • 'homework_session' parent trace with child spans");
    console.log("  • Token counts and cost per LLM call");
    console.log("  • Latency breakdown at each step");
    console.log("  • The error trace from the malformed prompt example");
  }
}

main().catch(console.error);
