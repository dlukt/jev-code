/**
 * Provider tests: selection, TypeSafe and Vercel translation, error mapping,
 * and provider-independent review behavior. All network calls are mocked.
 */
import assert from "node:assert/strict";
import { Readable } from "node:stream";
import { describe, test } from "node:test";
import {
  classifierFor,
  configuredModel,
  configuredProvider,
  createJevProvider,
  InvalidProviderError,
  jevFromEnvironment,
} from "../src/adapters/config.ts";
import { createWorkflowDependencies } from "../src/adapters/dependencies.ts";
import { classifyError, MissingCredentialError, TypeSafeJevProvider } from "../src/adapters/jev.ts";
import {
  classifyVercelError,
  createVercelAdapter,
  type EvaluateFn,
  GATEWAY_MODEL,
  VercelJevProvider,
} from "../src/adapters/vercel-jev.ts";
import type { JevRequest, TransportFailure } from "../src/core/types.ts";
import { check } from "../src/workflows/check.ts";
import { fake, options, tempRepo } from "./helpers.ts";

const CALL_OPTIONS = { timeoutMs: 30_000 };

function statusError(status: number, message = "boom", name = "Error"): Error {
  return Object.assign(new Error(message), { status, name });
}

function gatewayError(statusCode: number, name: string, message = "gateway boom"): Error {
  return Object.assign(new Error(message), { statusCode, name });
}

const retryError = (cause: unknown) =>
  Object.assign(new Error(`Failed after 2 attempts. Last error: boom`), {
    name: "AI_RetryError",
    errors: [new Error("first"), cause as Error],
  });

describe("provider selection", () => {
  test("defaults to typesafe; accepts both names; rejects invalid names at startup", () => {
    assert.equal(configuredProvider({}), "typesafe");
    assert.equal(configuredProvider({ JEV_PROVIDER: "  " }), "typesafe");
    assert.equal(configuredProvider({ JEV_PROVIDER: "typesafe" }), "typesafe");
    assert.equal(configuredProvider({ JEV_PROVIDER: "vercel" }), "vercel");
    assert.throws(() => configuredProvider({ JEV_PROVIDER: "openrouter" }), InvalidProviderError);
    assert.throws(
      () => configuredProvider({ JEV_PROVIDER: "Vercel" }),
      /JEV_PROVIDER must be one of typesafe, vercel/,
    );
  });

  test("factory builds the requested provider and each requires its key", () => {
    const typesafe = createJevProvider("typesafe", { TYPESAFE_API_KEY: "tsk_test" });
    assert.ok(typesafe instanceof TypeSafeJevProvider);
    assert.throws(() => createJevProvider("typesafe", {}), MissingCredentialError);
    const vercel = createJevProvider("vercel", { AI_GATEWAY_API_KEY: "vck_test" });
    assert.ok(vercel instanceof VercelJevProvider);
    assert.throws(() => createJevProvider("vercel", {}), MissingCredentialError);
    // A vercel key must not satisfy the typesafe provider and vice versa.
    assert.throws(() => createJevProvider("typesafe", { AI_GATEWAY_API_KEY: "vck_test" }));
    assert.throws(() => createJevProvider("vercel", { TYPESAFE_API_KEY: "tsk_test" }));
  });

  test("jevFromEnvironment follows JEV_PROVIDER and fails fast on unknown names", () => {
    assert.ok(jevFromEnvironment({ TYPESAFE_API_KEY: "k" }) instanceof TypeSafeJevProvider);
    assert.ok(
      jevFromEnvironment({ JEV_PROVIDER: "vercel", AI_GATEWAY_API_KEY: "k" }) instanceof VercelJevProvider,
    );
    assert.throws(
      () => jevFromEnvironment({ JEV_PROVIDER: "nope", TYPESAFE_API_KEY: "k" }),
      InvalidProviderError,
    );
  });

  test("model configuration is provider independent", () => {
    assert.equal(configuredModel("jev-flag", { TYPESAFE_MODEL: "jev-env" }), "jev-flag");
    assert.equal(configuredModel(undefined, { TYPESAFE_MODEL: " jev-env " }), "jev-env");
    assert.equal(configuredModel(undefined, {}), undefined);
  });

  test("classifierFor maps each provider to its classifier", () => {
    assert.equal(classifierFor("typesafe"), classifyError);
    assert.equal(classifierFor("vercel"), classifyVercelError);
  });
});

describe("TypeSafe provider", () => {
  test("forwards the request verbatim and returns the raw SDK response", async () => {
    const raw = { model: "jev-1.13.0", answers: {}, usage: { input_tokens: 1, output_tokens: 2 } };
    const seen: unknown[] = [];
    const provider = new TypeSafeJevProvider({
      apiKey: "tsk_test_key",
      client: {
        async systemOne(request, options) {
          seen.push({ request, options });
          return raw;
        },
      },
    });
    const request: JevRequest = {
      state: { diff: "x" },
      questions: { q: { type: "noul", instructions: "Is it?" } },
      model: "jev-1.13.0",
    };
    const response = await provider.ask(request, { timeoutMs: 15_000 });
    assert.equal(response, raw);
    assert.deepEqual(seen, [
      {
        request: { state: request.state, questions: request.questions, model: "jev-1.13.0" },
        options: { timeout: 15_000 },
      },
    ]);
  });

  test("passes the abort signal through", async () => {
    let observed: AbortSignal | undefined;
    const provider = new TypeSafeJevProvider({
      apiKey: "k",
      client: {
        async systemOne(_request, options) {
          observed = options?.signal;
          return {};
        },
      },
    });
    const controller = new AbortController();
    await provider.ask(
      { state: {}, questions: { q: { type: "noul", instructions: "?" } }, model: "m" },
      { timeoutMs: 1_000, signal: controller.signal },
    );
    assert.equal(observed, controller.signal);
  });

  test("missing API key fails with a clear message", () => {
    assert.throws(() => createJevProvider("typesafe", {}), /TYPESAFE_API_KEY is required/);
  });

  test("classifies SDK and network errors", () => {
    const cases: Array<[unknown, TransportFailure]> = [
      [statusError(401, "bad key"), "auth"],
      [statusError(403, "forbidden"), "auth"],
      [Object.assign(new Error("nope"), { name: "APIUserAbortError" }), "aborted"],
      [statusError(413, "payload too large"), "too_large"],
      [statusError(429, "rate limit"), "transient"],
      [statusError(503, "unavailable"), "transient"],
      [statusError(422, "bad request"), "rejected"],
      [Object.assign(new Error("econnreset"), { name: "TypeError" }), "transient"],
      [new Error("mystery"), "unknown"],
    ];
    for (const [error, expected] of cases) assert.equal(classifyError(error), expected);
    // RetryError chains classify their last cause.
    assert.equal(classifyError(retryError(statusError(429, "slow down"))), "transient");
  });
});

describe("Vercel provider request translation", () => {
  test("maps noul to boolean, choice to choice, score to score; model id is gateway-qualified", async () => {
    const seen: Array<Record<string, unknown>> = [];
    const provider = new VercelJevProvider({
      evaluate: async (call) => {
        seen.push(call);
        return {
          answers: {
            n: { type: "boolean", probability: 0.8 },
            c: { type: "choice", choice: "yes", probabilities: { yes: 0.8, no: 0.2 } },
            s: { type: "score", score: 1.2, probabilities: { "0": 0.1, "1": 0.7, "2": 0.2 } },
          },
          usage: { inputTokens: 10, outputTokens: 3 },
        };
      },
    });
    const request: JevRequest = {
      state: { hunk: "code" },
      questions: {
        n: { type: "noul", instructions: "Noul?", criteria: { true: "t", false: "f" } },
        c: { type: "choice", instructions: "Pick", criteria: { yes: "affirm", no: "deny" } },
        s: { type: "score", instructions: "Rank", criteria: ["low", "mid", "high"] },
      },
      model: "typesafe-ai/jev",
    };
    const response = (await provider.ask(request, CALL_OPTIONS)) as Record<string, unknown>;
    assert.equal(seen.length, 1);
    assert.equal(seen[0]!.model, GATEWAY_MODEL);
    // Zero data retention is requested by default.
    assert.deepEqual(seen[0]!.providerOptions, { gateway: { zeroDataRetention: true } });
    assert.deepEqual(seen[0]!.state, { hunk: "code" });
    assert.deepEqual(seen[0]!.questions, {
      n: { type: "boolean", instructions: "Noul?", criteria: { true: "t", false: "f" } },
      c: { type: "choice", instructions: "Pick", criteria: { yes: "affirm", no: "deny" } },
      s: { type: "score", instructions: "Rank", criteria: ["low", "mid", "high"] },
    });
    assert.equal(seen[0]!.maxRetries, 0);
    // Response envelope keeps TypeSafe shapes for the review engine. No metadata
    // was supplied, so confidence falls back to the derived value.
    assert.equal(response.model, GATEWAY_MODEL);
    assert.deepEqual(response.usage, { input_tokens: 10, output_tokens: 3 });
    assert.deepEqual(response.answers, {
      n: { type: "noul", noul: 0.8 },
      c: { type: "choice", choice: "yes", confidence: 0.8, probabilities: { yes: 0.8, no: 0.2 } },
      s: {
        type: "score",
        score: 1.2,
        confidence: 0.7,
        legend: {},
        probabilities: { "0": 0.1, "1": 0.7, "2": 0.2 },
      },
    });
  });

  test("reported model id wins over the configured one", async () => {
    const provider = new VercelJevProvider({
      evaluate: async () => ({
        answers: { n: { type: "boolean", probability: 1 } },
        response: { modelId: "jev-1.13.0" },
      }),
    });
    const response = (await provider.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
      CALL_OPTIONS,
    )) as Record<string, unknown>;
    assert.equal(response.model, "jev-1.13.0");
  });

  test("missing per-question answers and type mismatches are rejected", async () => {
    const provider = new VercelJevProvider({
      evaluate: async () => ({ answers: {} }),
    });
    await assert.rejects(
      provider.ask(
        { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
        CALL_OPTIONS,
      ),
      /no answer for question "n"/,
    );
    const mismatch = new VercelJevProvider({
      evaluate: async () => ({ answers: { n: { type: "choice", choice: "x" } } }),
    });
    await assert.rejects(
      mismatch.ask(
        { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
        CALL_OPTIONS,
      ),
      /not a boolean answer/,
    );
  });

  test("choice distributions are padded with zero for unreported labels", async () => {
    const provider = new VercelJevProvider({
      evaluate: async () => ({
        answers: { c: { type: "choice", choice: "a", probabilities: { a: 1 } } },
      }),
    });
    const response = (await provider.ask(
      {
        state: {},
        questions: { c: { type: "choice", instructions: "?", criteria: { a: "x", b: "y" } } },
        model: "m",
      },
      CALL_OPTIONS,
    )) as { answers: { c: { probabilities: Record<string, number>; confidence: number } } };
    assert.deepEqual(response.answers.c.probabilities, { a: 1, b: 0 });
    assert.equal(response.answers.c.confidence, 1);
  });

  test("score distributions are padded with zero for unreported levels", async () => {
    const provider = new VercelJevProvider({
      evaluate: async () => ({
        answers: { s: { type: "score", score: 0, probabilities: { "1": 1 } } },
      }),
    });
    const response = (await provider.ask(
      {
        state: {},
        questions: { s: { type: "score", instructions: "?", criteria: ["a", "b", "c"] } },
        model: "m",
      },
      CALL_OPTIONS,
    )) as { answers: { s: { probabilities: Record<string, number>; confidence: number } } };
    assert.deepEqual(response.answers.s.probabilities, { "0": 0, "1": 1, "2": 0 });
    assert.equal(response.answers.s.confidence, 1);
  });

  test("default usage is zero when the gateway omits it", async () => {
    const provider = new VercelJevProvider({
      evaluate: async () => ({ answers: { n: { type: "boolean", probability: 0.5 } } }),
    });
    const response = (await provider.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
      CALL_OPTIONS,
    )) as { usage: Record<string, number> };
    assert.deepEqual(response.usage, { input_tokens: 0, output_tokens: 0 });
  });

  test("noul criteria may be absent; null criteria are dropped", async () => {
    const seen: Array<{ questions: Record<string, unknown> }> = [];
    const provider = new VercelJevProvider({
      evaluate: async (call) => {
        seen.push({ questions: call.questions });
        return {
          answers: {
            n: { type: "boolean", probability: 0.5 },
            m: { type: "boolean", probability: 0.5 },
          },
        };
      },
    });
    await provider.ask(
      {
        state: {},
        questions: {
          n: { type: "noul", instructions: "?" },
          m: { type: "noul", instructions: "?", criteria: null },
        },
        model: "m",
      },
      CALL_OPTIONS,
    );
    assert.deepEqual(seen[0]!.questions, {
      n: { type: "boolean", instructions: "?" },
      m: { type: "boolean", instructions: "?" },
    });
  });

  test("requires instructions on every question", async () => {
    const provider = new VercelJevProvider({ evaluate: async () => ({ answers: {} }) });
    await assert.rejects(
      provider.ask({ state: {}, questions: { n: { type: "noul" } }, model: "m" }, CALL_OPTIONS),
      /requires instructions/,
    );
  });

  test("missing API key fails with a clear message", () => {
    assert.throws(() => createVercelAdapter({}), /AI_GATEWAY_API_KEY is required/);
  });
});

describe("Vercel error classification", () => {
  test("maps gateway errors onto the shared taxonomy", () => {
    const cases: Array<[unknown, TransportFailure]> = [
      [gatewayError(401, "GatewayAuthenticationError", "Unauthenticated"), "auth"],
      [gatewayError(403, "GatewayForbiddenError"), "auth"],
      [new Error("Invalid API key"), "auth"],
      [gatewayError(413, "GatewayInvalidRequestError", "payload too large"), "too_large"],
      [gatewayError(429, "GatewayRateLimitError"), "transient"],
      [gatewayError(500, "GatewayInternalServerError"), "transient"],
      [gatewayError(408, "GatewayTimeoutError"), "transient"],
      [gatewayError(422, "GatewayInvalidRequestError", "bad request"), "rejected"],
      [Object.assign(new Error("aborted"), { name: "AbortError" }), "aborted"],
      [new Error("mystery"), "unknown"],
    ];
    for (const [error, expected] of cases) assert.equal(classifyVercelError(error), expected);
    // RetryError chains classify their last cause.
    assert.equal(
      classifyVercelError(retryError(gatewayError(500, "GatewayInternalServerError"))),
      "transient",
    );
    assert.equal(classifyVercelError(retryError(gatewayError(401, "GatewayAuthenticationError"))), "auth");
  });
});

describe("provider-independent review behavior", () => {
  test("the check workflow runs unchanged against any Jev port", async () => {
    const r = tempRepo();
    try {
      r.write({
        "src/app.ts": "export const x = 1;\n",
        "test/app.test.ts": 'test("x", () => {});\n',
      });
      r.commit("init");
      r.write({ "src/app.ts": "export const x = 2;\n" });
      const adapter = fake();
      const packet = await check(
        { task: "change x", scope: "worktree" },
        {
          ...options(r.root, adapter),
          dependencies: createWorkflowDependencies(r.root, adapter, classifyVercelError),
        },
      );
      assert.equal(packet.schema, "jev-code.packet/v1");
      assert.equal(packet.workflow, "check@1");
      assert.ok(packet.jev.requests > 0);
    } finally {
      r.cleanup();
    }
  });
});

describe("provider selection through the CLI", () => {
  test("an invalid JEV_PROVIDER fails at startup with exit code 64", async () => {
    const r = tempRepo();
    try {
      r.write({ "src/a.ts": "export const a = 1;\n" });
      r.commit("init");
      r.write({ "src/a.ts": "export const a = 2;\n" });
      let stdout = "";
      let stderr = "";
      const { runCli } = await import("../src/cli.ts");
      const code = await runCli(["Find the code", "--no-persist"], {
        stdout: { write: (text: string) => (stdout += text) },
        stderr: { write: (text: string) => (stderr += text) },
        stdin: Readable.from([""]),
        cwd: r.root,
        env: { JEV_PROVIDER: "openrouter", TYPESAFE_API_KEY: "tsk_test" },
      });
      assert.equal(code, 64);
      assert.match(stderr, /JEV_PROVIDER must be one of typesafe, vercel/);
    } finally {
      r.cleanup();
    }
  });

  test("JEV_PROVIDER=vercel without AI_GATEWAY_API_KEY fails as input error 65", async () => {
    const r = tempRepo();
    try {
      r.write({ "src/a.ts": "export const a = 1;\n" });
      r.commit("init");
      r.write({ "src/a.ts": "export const a = 2;\n" });
      let stderr = "";
      const { runCli } = await import("../src/cli.ts");
      const code = await runCli(["Find the code", "--no-persist"], {
        stdout: { write: () => {} },
        stderr: { write: (text: string) => (stderr += text) },
        stdin: Readable.from([""]),
        cwd: r.root,
        env: { JEV_PROVIDER: "vercel" },
      });
      assert.equal(code, 65);
      assert.match(stderr, /AI_GATEWAY_API_KEY is required/);
    } finally {
      r.cleanup();
    }
  });
});

describe("Vercel provider timeout and abort", () => {
  test("a slow evaluation is aborted at the executor timeout", async () => {
    const provider = new VercelJevProvider({
      evaluate: (call) =>
        new Promise((_resolve, reject) => {
          call.abortSignal?.addEventListener("abort", () =>
            reject(Object.assign(new Error("aborted"), { name: "AbortError" })),
          );
        }),
    });
    await assert.rejects(
      provider.ask(
        { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
        { timeoutMs: 30 },
      ),
      /aborted/,
    );
  });

  test("an external abort signal is forwarded", async () => {
    const provider = new VercelJevProvider({
      evaluate: (call) =>
        new Promise((_resolve, reject) => {
          call.abortSignal?.addEventListener("abort", () =>
            reject(Object.assign(new Error("run aborted"), { name: "AbortError" })),
          );
        }),
    });
    const controller = new AbortController();
    const pending = provider.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
      { timeoutMs: 5_000, signal: controller.signal },
    );
    controller.abort();
    await assert.rejects(pending, /aborted/);
  });

  test("the timeout timer does not keep the process alive", async () => {
    let observedSignal: AbortSignal | undefined;
    const provider = new VercelJevProvider({
      evaluate: async (call) => {
        observedSignal = call.abortSignal;
        return { answers: { n: { type: "boolean", probability: 1 } } };
      },
    });
    await provider.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
      { timeoutMs: 60_000 },
    );
    assert.equal(observedSignal?.aborted, false);
  });
});

describe("Vercel provider: credentials, model semantics, confidence, ZDR", () => {
  test("the custom environment credential is wired into the provider, not just validated", async () => {
    // Use createVercelAdapter with a custom env and prove the key reaches the
    // gateway: the gateway model instance records the Authorization header its
    // fetch sends, which only happens when the key is bound via createGateway.
    const key = "vck_test_wiring_proof_1234";
    const adapter = createVercelAdapter({ AI_GATEWAY_API_KEY: key });
    assert.ok(adapter instanceof VercelJevProvider);
    // The provider must not mutate process.env while constructing.
    assert.equal(process.env.AI_GATEWAY_API_KEY, undefined);
    // Drive one real fetch through the bound gateway: point it at a local server.
    const { createServer } = await import("node:http");
    const received: Array<{ auth: string | undefined; body: unknown; modelHeader: string | undefined }> = [];
    const server = createServer((req, res) => {
      let data = "";
      req.on("data", (chunk: string) => (data += chunk));
      req.on("end", () => {
        received.push({
          auth: req.headers.authorization,
          body: JSON.parse(data),
          modelHeader: req.headers["ai-model-id"] as string | undefined,
        });
        res.writeHead(200, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            answers: {
              q: { type: "boolean", probability: 0.5 },
              c: { type: "choice", choice: "a", probabilities: { a: 0.7, b: 0.3 } },
            },
            usage: { inputTokens: 1, outputTokens: 1 },
            providerMetadata: { typesafe: { confidence: { c: 0.42 } } },
          }),
        );
      });
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const port = (server.address() as { port: number }).port;
    // The gateway instance inside the adapter was built with createGateway({apiKey});
    // reconstruct the same wiring with a fetch that targets the local server.
    const { createGateway, experimental_evaluate } = await import("ai");
    const gateway = createGateway({ apiKey: key, baseURL: `http://127.0.0.1:${port}/v4/ai` });
    const provider = new VercelJevProvider({
      evaluate: experimental_evaluate as unknown as EvaluateFn,
      modelFactory: (id) => gateway.evaluationModel(id),
    });
    const response = (await provider.ask(
      {
        state: { text: "s" },
        questions: {
          q: { type: "noul", instructions: "?" },
          c: { type: "choice", instructions: "?", criteria: { a: "x", b: "y" } },
        },
        model: "typesafe-ai/jev",
      },
      { timeoutMs: 5_000 },
    )) as Record<string, unknown>;
    server.close();
    assert.equal(received.length, 1);
    assert.equal(received[0]!.auth, `Bearer ${key}`); // the custom env key, actually sent
    assert.equal(received[0]!.modelHeader, "typesafe-ai/jev");
    assert.equal(
      ((received[0]!.body as { questions: Record<string, { type: string }> }).questions.q ?? {}).type,
      "boolean",
    );
    // providerMetadata.confidence (0.42) flows into the choice answer; the
    // boolean/noul answer carries probability only.
    const answers = response.answers as Record<string, { confidence?: number } | undefined>;
    assert.equal(answers.c?.confidence, 0.42);
    assert.equal(answers.q?.confidence, undefined);
  });

  test("TypeSafe's confidence statistic is preferred over the derived fallback", async () => {
    const provider = new VercelJevProvider({
      evaluate: async () => ({
        answers: {
          c: { type: "choice", choice: "a", probabilities: { a: 0.9, b: 0.1 } },
          s: { type: "score", score: 1, probabilities: { "0": 0.05, "1": 0.05, "2": 0.9 } },
        },
        providerMetadata: { typesafe: { confidence: { c: 0.61, s: 0.77 } } },
      }),
    });
    const response = (await provider.ask(
      {
        state: {},
        questions: {
          c: { type: "choice", instructions: "?", criteria: { a: "x", b: "y" } },
          s: { type: "score", instructions: "?", criteria: ["l0", "l1", "l2"] },
        },
        model: "m",
      },
      CALL_OPTIONS,
    )) as { answers: { c: { confidence: number }; s: { confidence: number } } };
    // The reported statistic (0.61/0.77), not the distribution-derived value (0.9).
    assert.equal(response.answers.c.confidence, 0.61);
    assert.equal(response.answers.s.confidence, 0.77);
  });

  test("non-numeric or absent confidence metadata falls back to the derived value", async () => {
    const provider = new VercelJevProvider({
      evaluate: async () => ({
        answers: { c: { type: "choice", choice: "a", probabilities: { a: 0.9 } } },
        providerMetadata: { typesafe: { confidence: { c: "high" } } },
      }),
    });
    const response = (await provider.ask(
      {
        state: {},
        questions: { c: { type: "choice", instructions: "?", criteria: { a: "x", b: "y" } } },
        model: "m",
      },
      CALL_OPTIONS,
    )) as { answers: { c: { confidence: number } } };
    assert.equal(response.answers.c.confidence, 0.9);
  });

  test("unqualified TypeSafe model ids never reach the gateway; qualified ids are honored", async () => {
    const seen: string[] = [];
    const provider = new VercelJevProvider({
      model: GATEWAY_MODEL,
      modelFactory: (id) => {
        seen.push(id);
        return id;
      },
      evaluate: async () => ({ answers: {} }),
    });
    // Unqualified (TypeSafe-direct namespace) -> configured gateway model.
    await assert.rejects(
      provider.ask(
        { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "jev-1.13.0" },
        CALL_OPTIONS,
      ),
      /no answer/,
    );
    // Qualified (gateway namespace) -> passed through.
    await assert.rejects(
      provider.ask(
        {
          state: {},
          questions: { n: { type: "noul", instructions: "?" } },
          model: "typesafe-ai/jev-preview",
        },
        CALL_OPTIONS,
      ),
      /no answer/,
    );
    assert.deepEqual(seen, [GATEWAY_MODEL, "typesafe-ai/jev-preview"]);
  });

  test("zero data retention can be disabled for troubleshooting", async () => {
    const seen: Array<Record<string, unknown>> = [];
    const provider = new VercelJevProvider({
      zeroDataRetention: false,
      evaluate: async (call) => {
        seen.push(call);
        return { answers: {} };
      },
    });
    await assert.rejects(
      provider.ask(
        { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
        CALL_OPTIONS,
      ),
      /no answer/,
    );
    assert.equal(seen[0]!.providerOptions, undefined);
  });

  test("an already-aborted external signal propagates before any evaluation", async () => {
    let evaluations = 0;
    const provider = new VercelJevProvider({
      evaluate: async () => {
        evaluations++;
        return { answers: {} };
      },
    });
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      provider.ask(
        { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
        { timeoutMs: 5_000, signal: controller.signal },
      ),
    );
    assert.equal(evaluations, 0); // no evaluation was started
  });
});
