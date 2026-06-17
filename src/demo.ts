/**
 * AUTOMATED DEMO — runs all features without manual input
 * -------------------------------------------------------
 * Run with: npm run demo
 * Shows chains, memory, and agent in one script — good for a quick showcase.
 */

import * as dotenv from "dotenv";
dotenv.config();

import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { getChatModel } from "./models";
import { allTools } from "./tools";
import { printLangSmithInfo, getConfig } from "./config";

function divider(title: string) {
  const line = "─".repeat(50);
  console.log(`\n${line}`);
  console.log(`  ${title}`);
  console.log(line);
}

async function demoChain() {
  divider("DEMO 1: Simple Chain");
  const model = getChatModel({ temperature: 0.5 });
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a concise study assistant. Answer in 2–3 sentences."],
    ["human", "{question}"],
  ]);
  const chain = prompt.pipe(model).pipe(new StringOutputParser());

  const answer = await chain.invoke({ question: "What is Big O notation?" });
  console.log(`Q: What is Big O notation?\nA: ${answer}`);
}

async function demoSequentialChain() {
  divider("DEMO 2: Sequential Chain (Explain → Quiz)");
  const model = getChatModel({ temperature: 0.6 });

  const explainChain = ChatPromptTemplate.fromMessages([
    ["system", "You are a teacher. Explain in 2 sentences."],
    ["human", "Explain: {topic}"],
  ]).pipe(model).pipe(new StringOutputParser());

  const quizChain = ChatPromptTemplate.fromMessages([
    ["system", "Create one quiz question based on this explanation."],
    ["human", "{explanation}"],
  ]).pipe(model).pipe(new StringOutputParser());

  const seq = RunnableSequence.from([
    explainChain,
    (explanation: string) => quizChain.invoke({ explanation }),
  ]);

  const quiz = await seq.invoke({ topic: "Hash tables" });
  console.log(`Topic: Hash tables\nGenerated Quiz Question:\n${quiz}`);
}

async function demoMemory() {
  divider("DEMO 3: Memory — Multi-turn Conversation");
  const model = getChatModel({ temperature: 0.5 });
  const sessionHistory = new ChatMessageHistory();

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a study assistant. Be brief and remember past context."],
    new MessagesPlaceholder("history"),
    ["human", "{input}"],
  ]);

  const chainWithHistory = new RunnableWithMessageHistory({
    runnable: prompt.pipe(model).pipe(new StringOutputParser()),
    getMessageHistory: () => sessionHistory,
    inputMessagesKey: "input",
    historyMessagesKey: "history",
  });

  const cfg = { configurable: { sessionId: "demo" } };

  const turns = [
    "I'm studying for a TypeScript exam.",
    "What topics should I focus on?",
    "Can you give me a quick tip for the first topic?",
  ];

  for (const turn of turns) {
    const response = await chainWithHistory.invoke({ input: turn }, cfg);
    console.log(`🧑: ${turn}`);
    console.log(`🤖: ${response}\n`);
  }
}

async function demoAgent() {
  divider("DEMO 4: Agent — Dynamic Tool Selection");
  const model = getChatModel({ temperature: 0 });

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a study assistant with tools. Use them when appropriate."],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"),
  ]);

  const agent = createToolCallingAgent({ llm: model, tools: allTools, prompt });
  const executor = new AgentExecutor({ agent, tools: allTools, verbose: false, maxIterations: 4 });

  const agentWithMemory = new RunnableWithMessageHistory({
    runnable: executor,
    getMessageHistory: () => new ChatMessageHistory(),
    inputMessagesKey: "input",
    historyMessagesKey: "chat_history",
    outputMessagesKey: "output",
  });

  const questions = [
    "What is 144 + 256?",
    "Explain recursion at beginner level.",
    "Give me study tips for programming.",
  ];

  for (const q of questions) {
    console.log(`🧑: ${q}`);
    const result = await agentWithMemory.invoke({ input: q }, { configurable: { sessionId: "agent-demo" } });
    console.log(`🤖: ${result.output}\n`);
  }
}

async function main() {
  const config = getConfig();

  console.log("╔══════════════════════════════════════════════════╗");
  console.log("║   LangChain TypeScript — Full Feature Demo        ║");
  console.log("╚══════════════════════════════════════════════════╝");

  printLangSmithInfo(config);

  await demoChain();
  await demoSequentialChain();
  await demoMemory();
  await demoAgent();

  console.log("\n" + "═".repeat(52));
  console.log("✅  All demos complete!");
  if (config.langsmithEnabled) {
    console.log(`📊 Traces at: https://smith.langchain.com (project: ${config.langsmithProject})`);
  }
  console.log("═".repeat(52));
}

main().catch(console.error);
