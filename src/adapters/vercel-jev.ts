import { experimental_evaluate as aiEvaluate } from "ai";
import type { Entry, Question, Questions } from "../core/questions.ts";
import type { JevCallOptions, JevPort, JevRequest, TransportFailure } from "../core/types.ts";
import { MissingCredentialError } from "./jev.ts";

export { MissingCredentialError };

/** Gateway model id for Jev on the Vercel AI Gateway. */
export const GATEWAY_MODEL = "typesafe-ai/jev";

/** Gateway evaluation questions: choice, score, and boolean (their name for noul). */
export type GatewayQuestion =
  | { type: "choice"; instructions: Entry; criteria: Record<string, Entry | null> }
  | { type: "score"; instructions: Entry; criteria: readonly (Entry | null)[] }
  | { type: "boolean"; instructions: Entry; criteria?: { true?: Entry | null; false?: Entry | null } };

export type GatewayAnswer =
  | { type: "choice"; choice: string; probabilities?: Record<string, number> }
  | { type: "score"; score: number; probabilities?: Record<string, number> }
  | { type: "boolean"; probability: number };

/** The slice of `evaluate()` from the `ai` package that the provider needs. */
export type EvaluateFn = (options: {
  model: string;
  state: Entry;
  questions: Record<string, GatewayQuestion>;
  maxRetries: number;
  abortSignal?: AbortSignal;
}) => Promise<{
  answers: Record<string, GatewayAnswer>;
  usage?: { inputTokens?: number; outputTokens?: number };
  response?: { modelId?: string };
}>;

export interface VercelJevProviderOptions {
  /** Performs one evaluation; defaults to `evaluate` from the `ai` package. */
  evaluate?: EvaluateFn;
  /** Gateway model id. */
  model?: string;
}

/**
 * Vercel AI Gateway Jev provider: one Jev port backed by the gateway evaluation
 * API (`typesafe-ai/jev`) instead of the TypeSafe API. Authentication stays in
 * the `ai` package, which reads AI_GATEWAY_API_KEY from the environment.
 */
export class VercelJevProvider implements JevPort {
  private readonly evaluate: EvaluateFn;
  private readonly model: string;

  constructor(options: VercelJevProviderOptions = {}) {
    this.evaluate = options.evaluate ?? (aiEvaluate as unknown as EvaluateFn);
    this.model = options.model ?? GATEWAY_MODEL;
  }

  async ask(request: JevRequest, options: JevCallOptions): Promise<unknown> {
    const result = await this.evaluate({
      model: this.model,
      state: request.state as Entry,
      questions: translateQuestions(request.questions),
      maxRetries: 0,
      ...(options.signal ? { abortSignal: options.signal } : {}),
    });
    return {
      model: result.response?.modelId ?? this.model,
      answers: translateAnswers(request.questions, result.answers),
      usage: {
        input_tokens: result.usage?.inputTokens ?? 0,
        output_tokens: result.usage?.outputTokens ?? 0,
      },
    };
  }
}

/** Build the gateway-backed Jev port. The API key is read only from the given environment. */
export function createVercelAdapter(env: NodeJS.ProcessEnv = process.env): JevPort {
  const apiKey = env.AI_GATEWAY_API_KEY?.trim();
  if (!apiKey)
    throw new MissingCredentialError(
      "AI_GATEWAY_API_KEY is required; set it in the process environment before running jev-code with JEV_PROVIDER=vercel",
    );
  return new VercelJevProvider();
}

/** Translate jev questions to gateway evaluation questions. */
export function translateQuestions(questions: Questions): Record<string, GatewayQuestion> {
  const out: Record<string, GatewayQuestion> = {};
  for (const [name, question] of Object.entries(questions)) out[name] = translateQuestion(question);
  return out;
}

function translateQuestion(question: Question): GatewayQuestion {
  if (question.type === "noul") {
    if (question.instructions === undefined) {
      throw new Error("the Vercel provider requires instructions on every noul question");
    }
    const criteria =
      question.criteria === undefined || question.criteria === null ? undefined : { ...question.criteria };
    return {
      type: "boolean",
      instructions: question.instructions,
      ...(criteria ? { criteria } : {}),
    };
  }
  if (question.type === "choice") {
    if (question.instructions === undefined) {
      throw new Error("the Vercel provider requires instructions on every choice question");
    }
    return { type: "choice", instructions: question.instructions, criteria: question.criteria };
  }
  if (question.instructions === undefined) {
    throw new Error("the Vercel provider requires instructions on every score question");
  }
  return { type: "score", instructions: question.instructions, criteria: question.criteria };
}

/**
 * Translate gateway answers into the answer shape the review engine validates.
 * The gateway reports no per-answer confidence; it is derived losslessly from
 * the distribution: the mass of the selected choice, or the maximum level mass
 * for score. Noul maps to boolean probability.
 */
export function translateAnswers(
  questions: Questions,
  answers: Record<string, GatewayAnswer>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [name, question] of Object.entries(questions)) {
    const answer = answers[name];
    if (!answer) throw new Error(`the gateway returned no answer for question "${name}"`);
    if (question.type === "noul") {
      if (answer.type !== "boolean") throw new Error(`answer "${name}" is not a boolean answer`);
      out[name] = { type: "noul", noul: answer.probability };
    } else if (question.type === "choice") {
      if (answer.type !== "choice") throw new Error(`answer "${name}" is not a choice answer`);
      const probabilities = answer.probabilities ?? {};
      const labels = Object.keys(question.criteria);
      out[name] = {
        type: "choice",
        choice: answer.choice,
        confidence: probabilities[answer.choice] ?? 0,
        probabilities: Object.fromEntries(labels.map((label) => [label, probabilities[label] ?? 0])),
      };
    } else {
      if (answer.type !== "score") throw new Error(`answer "${name}" is not a score answer`);
      const probabilities = answer.probabilities ?? {};
      const values = Array.from(
        { length: question.criteria.length },
        (_, index) => probabilities[String(index)] ?? 0,
      );
      out[name] = {
        type: "score",
        score: answer.score,
        confidence: Math.max(...values),
        legend: {},
        probabilities: Object.fromEntries(values.map((value, index) => [String(index), value])),
      };
    }
  }
  return out;
}

/** Map gateway and network errors onto transport failure classes. */
export function classifyVercelError(error: unknown): TransportFailure {
  const value = typeof error === "object" && error !== null ? (error as Record<string, unknown>) : {};
  // RetryError carries every attempt; classify the last cause. The provider asks
  // for maxRetries 0, so this only fires when callers enable SDK retries.
  const attempts = value.errors;
  if (Array.isArray(attempts) && attempts.length > 0) {
    return classifyVercelError(attempts[attempts.length - 1]);
  }
  const status = typeof value.statusCode === "number" ? value.statusCode : null;
  const name = typeof value.name === "string" ? value.name : "";
  const message = error instanceof Error ? error.message.toLowerCase() : String(error).toLowerCase();
  if (name === "AbortError" || name === "APIUserAbortError") return "aborted";
  if (
    status === 401 ||
    status === 403 ||
    /authentication|permissiondenied/i.test(name) ||
    /invalid api key|unauthenticated/i.test(message)
  ) {
    return "auth";
  }
  if (status === 413 || /max_tokens_exceeded|too large|payload|context length/.test(message))
    return "too_large";
  if (
    status === 408 ||
    status === 429 ||
    (status !== null && status >= 500) ||
    /timeout|connection|ratelimit/i.test(name) ||
    /econnreset|etimedout|socket hang up|rate limit/.test(message)
  ) {
    return "transient";
  }
  if (status !== null && status >= 400) return "rejected";
  return "unknown";
}
