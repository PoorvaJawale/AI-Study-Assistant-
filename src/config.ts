import * as dotenv from "dotenv";
dotenv.config();

export function getConfig() {
  const provider = process.env.MODEL_PROVIDER || "openai";

  if (provider === "anthropic") {
    if (!process.env.ANTHROPIC_API_KEY) {
      throw new Error("ANTHROPIC_API_KEY is required when MODEL_PROVIDER=anthropic");
    }
  } else {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY is required. Copy .env.example to .env and fill in your keys.");
    }
  }

  const langsmithEnabled = process.env.LANGCHAIN_TRACING_V2 === "true";
  if (langsmithEnabled && !process.env.LANGCHAIN_API_KEY) {
    console.warn("⚠️  LANGCHAIN_API_KEY not set — LangSmith tracing is disabled.");
  }

  return {
    provider,
    langsmithEnabled,
    langsmithProject: process.env.LANGCHAIN_PROJECT || "langchain-ts-assistant",
  };
}

export function printLangSmithInfo(config: ReturnType<typeof getConfig>) {
  if (config.langsmithEnabled) {
    console.log(`\n🔍 LangSmith tracing ACTIVE → project: "${config.langsmithProject}"`);
    console.log(`   View traces at: https://smith.langchain.com\n`);
  } else {
    console.log("\n💡 Tip: Set LANGCHAIN_TRACING_V2=true in .env to enable LangSmith tracing.\n");
  }
}
