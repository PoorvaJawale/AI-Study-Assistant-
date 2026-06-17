import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { getConfig } from "./config";

/**
 * Returns the chat model based on MODEL_PROVIDER env variable.
 * Supports OpenAI (default) and Anthropic Claude.
 */
export function getChatModel(options: { temperature?: number; streaming?: boolean } = {}): BaseChatModel {
  const config = getConfig();
  const { temperature = 0.7, streaming = false } = options;

  if (config.provider === "anthropic") {
    return new ChatAnthropic({
      model: "claude-3-5-sonnet-20241022",
      temperature,
      streaming,
    });
  }

  return new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature,
    streaming,
  });
}
