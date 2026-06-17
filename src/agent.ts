/**
 * AGENT DEMO
 * ----------
 * An agent uses an LLM as a "reasoning engine" to decide WHICH tools to call
 * and in what order, based on the user's question.
 *
 * Unlike chains (fixed flow), agents are dynamic — the model decides at runtime.
 *
 * Architecture:
 *   User input → Agent (LLM + tools) → Tool calls → Observation → Final answer
 *
 * LangSmith shows you every reasoning step, tool call, and token used.
 */

import * as dotenv from "dotenv";
dotenv.config();

import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { getChatModel } from "./models";
import { allTools } from "./tools";
import { printLangSmithInfo, getConfig } from "./config";

const sessionStore: Record<string, ChatMessageHistory> = {};

function getSessionHistory(sessionId: string): ChatMessageHistory {
  if (!sessionStore[sessionId]) {
    sessionStore[sessionId] = new ChatMessageHistory();
  }
  return sessionStore[sessionId];
}

async function createStudyAgent() {
  const model = getChatModel({ temperature: 0 }); // temperature=0 for consistent tool decisions

  // Agent prompt: tells the LLM its role and how to behave
  const agentPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an expert Study Assistant AI. Your goal is to help students learn effectively.

You have access to the following tools:
- calculator: For any math calculations
- topic_explainer: For retrieving CS/programming topic explanations (use this FIRST before generating your own)
- study_tips: For giving subject-specific study advice

Guidelines:
1. Always use the topic_explainer tool before explaining a CS topic
2. Use the calculator tool for any numerical calculations — never compute in your head
3. Be encouraging and clear in your explanations
4. If a student seems stuck, offer study tips

Current date: ${new Date().toLocaleDateString()}`,
    ],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"), // where the agent writes its reasoning
  ]);

  // Create a tool-calling agent (uses function-calling API under the hood)
  const agent = createToolCallingAgent({
    llm: model,
    tools: allTools,
    prompt: agentPrompt,
  });

  // AgentExecutor runs the agent loop: reason → call tool → observe → repeat until done
  const executor = new AgentExecutor({
    agent,
    tools: allTools,
    verbose: true,       // logs each step to console
    maxIterations: 5,    // safety limit on reasoning loops
    returnIntermediateSteps: true, // include tool calls in the response
  });

  return executor;
}

async function runAgentDemo() {
  const executor = await createStudyAgent();

  // Wrap with memory so the agent remembers prior conversation turns
  const agentWithMemory = new RunnableWithMessageHistory({
    runnable: executor,
    getMessageHistory: getSessionHistory,
    inputMessagesKey: "input",
    historyMessagesKey: "chat_history",
    outputMessagesKey: "output",
  });

  const config = { configurable: { sessionId: "study-agent-demo" } };

  const questions = [
    "Hi! Can you explain binary search at an intermediate level?",
    "What is 2^10 + 512?",
    "Give me some study tips for programming.",
    "How is binary search different from linear search? Which is faster for 1 million items?",
  ];

  for (const question of questions) {
    console.log(`\n${"═".repeat(60)}`);
    console.log(`🧑 Student: ${question}`);
    console.log("═".repeat(60));

    const result = await agentWithMemory.invoke({ input: question }, config);

    console.log(`\n🤖 Assistant: ${result.output}`);

    // Show which tools were used (if any)
    if (result.intermediateSteps?.length > 0) {
      console.log("\n🔧 Tools Used:");
      for (const step of result.intermediateSteps) {
        console.log(`   → ${step.action.tool}(${JSON.stringify(step.action.toolInput)})`);
        console.log(`     Result: ${step.observation}`);
      }
    }
  }
}

async function main() {
  const config = getConfig();
  printLangSmithInfo(config);

  console.log("🤖 LangChain Agent Demo — Study Assistant\n");
  console.log("Agents differ from chains: the LLM decides WHICH tools to call at runtime.");
  console.log("Watch the 'Entering new AgentExecutor chain' logs to see the reasoning steps.\n");

  await runAgentDemo();

  console.log("\n✅ Agent demo complete.");
  console.log("🔍 In LangSmith, you can see:");
  console.log("   • Every LLM call and its input/output tokens");
  console.log("   • Each tool invocation and its result");
  console.log("   • The full reasoning chain from question to answer");
  console.log("   • Latency for each step");
}

main().catch(console.error);
