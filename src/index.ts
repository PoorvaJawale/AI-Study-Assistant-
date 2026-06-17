/**
 * MAIN ENTRY — Interactive Study Assistant
 * =========================================
 * Combines chains + memory + agent into a single conversational app.
 * Run with: npm run dev
 *
 * This is the "glue" that puts everything together for a real use-case:
 * a multi-turn study chatbot that remembers context and uses tools.
 */

import * as dotenv from "dotenv";
dotenv.config();

import * as readline from "readline";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { getChatModel } from "./models";
import { allTools } from "./tools";
import { printLangSmithInfo, getConfig } from "./config";

// ── Session Store ─────────────────────────────────────────────────────────────
const sessionStore: Record<string, ChatMessageHistory> = {};
function getSessionHistory(sessionId: string): ChatMessageHistory {
  if (!sessionStore[sessionId]) {
    sessionStore[sessionId] = new ChatMessageHistory();
  }
  return sessionStore[sessionId];
}

// ── Build the Agent ───────────────────────────────────────────────────────────
async function buildAssistant() {
  const model = getChatModel({ temperature: 0.3 });

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are StudyBot — an intelligent, encouraging AI study assistant.

Your capabilities:
- Explain programming and CS concepts clearly at any difficulty level
- Solve math problems using your calculator tool
- Give personalized study tips
- Remember what the student has asked earlier in this session

Available tools: calculator, topic_explainer, study_tips
Always use the topic_explainer tool before generating CS explanations.

Be concise, clear, and supportive. Use emojis sparingly to make responses friendly.`,
    ],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"),
  ]);

  const agent = createToolCallingAgent({ llm: model, tools: allTools, prompt });

  const executor = new AgentExecutor({
    agent,
    tools: allTools,
    verbose: false,
    maxIterations: 5,
  });

  return new RunnableWithMessageHistory({
    runnable: executor,
    getMessageHistory: getSessionHistory,
    inputMessagesKey: "input",
    historyMessagesKey: "chat_history",
    outputMessagesKey: "output",
  });
}

// ── Interactive Chat Loop ─────────────────────────────────────────────────────
async function main() {
  const config = getConfig();

  console.clear();
  console.log("╔══════════════════════════════════════════════════╗");
  console.log("║        📚 StudyBot — AI Study Assistant          ║");
  console.log("║   Powered by LangChain + TypeScript              ║");
  console.log("╚══════════════════════════════════════════════════╝");

  printLangSmithInfo(config);

  console.log("I can help you with:");
  console.log("  • Explaining programming & CS concepts");
  console.log("  • Solving math problems");
  console.log("  • Creating study plans");
  console.log("  • Giving study tips\n");
  console.log("Type 'exit' to quit, 'history' to see chat history, 'clear' to reset.\n");

  const assistant = await buildAssistant();
  const sessionId = `session-${Date.now()}`;
  const runConfig = { configurable: { sessionId } };

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const askQuestion = () => {
    rl.question("🧑 You: ", async (input: string) => {
      const trimmed = input.trim();

      if (!trimmed) {
        askQuestion();
        return;
      }

      if (trimmed.toLowerCase() === "exit") {
        console.log("\n👋 Goodbye! Keep studying hard!\n");
        rl.close();
        return;
      }

      if (trimmed.toLowerCase() === "history") {
        const history = getSessionHistory(sessionId);
        const messages = await history.getMessages();
        console.log(`\n📝 Chat History (${messages.length} messages):`);
        messages.forEach((msg, i) => {
          const role = msg._getType() === "human" ? "🧑 You" : "🤖 Bot";
          const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
          console.log(`  ${i + 1}. ${role}: ${content.substring(0, 100)}${content.length > 100 ? "..." : ""}`);
        });
        console.log();
        askQuestion();
        return;
      }

      if (trimmed.toLowerCase() === "clear") {
        delete sessionStore[sessionId];
        console.log("🗑️  Chat history cleared.\n");
        askQuestion();
        return;
      }

      try {
        process.stdout.write("🤖 StudyBot: ");
        const result = await assistant.invoke({ input: trimmed }, runConfig);
        console.log(result.output);
        console.log();
      } catch (error) {
        const err = error as Error;
        console.error("❌ Error:", err.message);
        if (config.langsmithEnabled) {
          console.log("   Check LangSmith for the full error trace.");
        }
        console.log();
      }

      askQuestion();
    });
  };

  askQuestion();
}

main().catch((err) => {
  console.error("Fatal error:", err.message);
  process.exit(1);
});
