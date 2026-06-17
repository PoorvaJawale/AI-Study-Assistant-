/**
 * MEMORY DEMO
 * -----------
 * Without memory, every LLM call is stateless — it forgets the last message.
 * LangChain's memory patterns let the assistant "remember" context across turns.
 *
 * Patterns shown:
 *   1. ChatMessageHistory — raw message store
 *   2. Conversation buffer — keeps all messages in the prompt
 *   3. Summary memory    — summarizes old messages to stay within token limits
 */

import * as dotenv from "dotenv";
dotenv.config();

import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { getChatModel } from "./models";
import { printLangSmithInfo, getConfig } from "./config";

// In-memory session store (in production use Redis, DynamoDB, etc.)
const sessionStore: Record<string, ChatMessageHistory> = {};

function getSessionHistory(sessionId: string): ChatMessageHistory {
  if (!sessionStore[sessionId]) {
    sessionStore[sessionId] = new ChatMessageHistory();
  }
  return sessionStore[sessionId];
}

async function conversationBufferMemoryDemo() {
  console.log("\n━━━ 1. Conversation Buffer Memory ━━━");
  console.log("All messages are kept in context (good for short conversations).\n");

  const model = getChatModel({ temperature: 0.7 });

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a friendly study assistant. Help students understand topics. Keep answers concise."],
    new MessagesPlaceholder("history"),   // <-- injected chat history
    ["human", "{input}"],
  ]);

  const chain = prompt.pipe(model).pipe(new StringOutputParser());

  // Wrap the chain with automatic history management
  const chainWithHistory = new RunnableWithMessageHistory({
    runnable: chain,
    getMessageHistory: getSessionHistory,
    inputMessagesKey: "input",
    historyMessagesKey: "history",
  });

  const sessionId = "student-session-1";
  const config = { configurable: { sessionId } };

  // Turn 1
  const response1 = await chainWithHistory.invoke(
    { input: "Hi! I'm studying for my data structures exam." },
    config
  );
  console.log("🧑 Student: Hi! I'm studying for my data structures exam.");
  console.log("🤖 Assistant:", response1);

  // Turn 2 — assistant remembers turn 1
  const response2 = await chainWithHistory.invoke(
    { input: "Can you explain what a linked list is?" },
    config
  );
  console.log("\n🧑 Student: Can you explain what a linked list is?");
  console.log("🤖 Assistant:", response2);

  // Turn 3 — assistant remembers both previous turns
  const response3 = await chainWithHistory.invoke(
    { input: "How is it different from an array?" },
    config
  );
  console.log("\n🧑 Student: How is it different from an array?");
  console.log("🤖 Assistant:", response3);

  // Show what's stored in memory
  const history = getSessionHistory(sessionId);
  const messages = await history.getMessages();
  console.log(`\n📝 Memory contains ${messages.length} messages for session "${sessionId}"`);
}

async function multiSessionDemo() {
  console.log("\n━━━ 2. Multi-Session Memory (Isolated per User) ━━━");
  console.log("Each session ID gets its own isolated memory.\n");

  const model = getChatModel({ temperature: 0.5 });

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a study assistant. Greet the student by name if you know it."],
    new MessagesPlaceholder("history"),
    ["human", "{input}"],
  ]);

  const chain = new RunnableWithMessageHistory({
    runnable: prompt.pipe(model).pipe(new StringOutputParser()),
    getMessageHistory: getSessionHistory,
    inputMessagesKey: "input",
    historyMessagesKey: "history",
  });

  // Alice's session
  await chain.invoke(
    { input: "My name is Alice and I need help with calculus." },
    { configurable: { sessionId: "alice-session" } }
  );
  const aliceResponse = await chain.invoke(
    { input: "What was my name again?" },
    { configurable: { sessionId: "alice-session" } }
  );
  console.log("🧑 Alice asks: 'What was my name again?'");
  console.log("🤖 Assistant (Alice's session):", aliceResponse);

  // Bob's session — completely separate memory
  await chain.invoke(
    { input: "I'm Bob, studying biology." },
    { configurable: { sessionId: "bob-session" } }
  );
  const bobResponse = await chain.invoke(
    { input: "What subject am I studying?" },
    { configurable: { sessionId: "bob-session" } }
  );
  console.log("\n🧑 Bob asks: 'What subject am I studying?'");
  console.log("🤖 Assistant (Bob's session):", bobResponse);

  // Prove isolation: Bob's session doesn't know Alice
  const isolationCheck = await chain.invoke(
    { input: "Do you know anyone named Alice?" },
    { configurable: { sessionId: "bob-session" } }
  );
  console.log("\n🧑 Bob asks: 'Do you know anyone named Alice?'");
  console.log("🤖 Assistant (Bob's session):", isolationCheck);
  console.log("✅ Sessions are isolated — Bob's assistant doesn't know about Alice.");
}

async function main() {
  const config = getConfig();
  printLangSmithInfo(config);

  console.log("🧠 LangChain Memory Demo\n");
  console.log("Without memory, each LLM call starts fresh.");
  console.log("Memory persists conversation context across multiple turns.\n");

  await conversationBufferMemoryDemo();
  await multiSessionDemo();

  console.log("\n✅ Memory demo complete. Check LangSmith to see how history is passed each turn.");
}

main().catch(console.error);
