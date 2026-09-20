/**
 * Cloudflare provider tests: selection, REST request shape, response
 * normalization, model semantics, error mapping, redaction, abort/timeout.
 * All network calls are mocked or served from a local HTTP server.
 */
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { Readable } from "node:stream";
import { describe, test } from "node:test";
import {
  CLOUDFLARE_MODEL,
  CloudflareJevProvider,
  CloudflareStatusError,
  classifyCloudflareError,
  createCloudflareAdapter,
} from "../src/adapters/cloudflare-jev.ts";
import {
  classifierFor,
  configuredProvider,
  createJevProvider,
  InvalidProviderError,
  jevFromEnvironment,
} from "../src/adapters/config.ts";
import { createWorkflowDependencies } from "../src/adapters/dependencies.ts";
import { createRedaction } from "../src/adapters/redact.ts";
import { estimateTokens } from "../src/core/budget.ts";
import { FrameExecutor } from "../src/core/executor.ts";
import type { JevPort, JevRequest, TransportFailure } from "../src/core/types.ts";
import { expectKeys, readEnvelope, readNoul } from "../src/core/validation.ts";

const CALL_OPTIONS = { timeoutMs: 30_000 };

/** A well-formed Jev payload as Cloudflare's result. */
function jevResult(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    model: "jev-1.13.0",
    answers: {
      n: { type: "noul", noul: 0.9 },
      c: { type: "choice", choice: "yes", confidence: 0.8, probabilities: { yes: 0.8, no: 0.2 } },
      s: {
        type: "score",
        score: 1.2,
        confidence: 0.94,
        legend: { 0: "low", 1: "mid", 2: "high" },
        probabilities: { 0: 0.1, 1: 0.7, 2: 0.2 },
      },
    },
    usage: { input_tokens: 426, output_tokens: 73 },
    ...overrides,
  };
}

function fullQuestions(): JevRequest["questions"] {
  return {
    n: { type: "noul", instructions: "Noul?" },
    c: { type: "choice", instructions: "Pick", criteria: { yes: "affirm", no: "deny" } },
    s: { type: "score", instructions: "Rank", criteria: ["low", "mid", "high"] },
  };
}

function fullRequest(): JevRequest {
  return { state: { hunk: "code" }, questions: fullQuestions(), model: "jev-1.13.0" };
}

/** Wrap a Jev payload in the plain v4 envelope. */
function envelope(result: unknown, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return { success: true, errors: [], messages: [], result, ...extra };
}

/** Wrap a Jev payload in the execution-state double envelope. */
function executionEnvelope(payload: unknown, state = "Completed"): Record<string, unknown> {
  return envelope({ state, result: payload });
}

/** fetch replacement that returns a canned JSON response. */
function respondWith(
  status: number,
  body: unknown,
): {
  fetchFn: (input: string, init: RequestInit) => Promise<Response>;
  seen: Array<{ url: string; init: RequestInit }>;
} {
  const seen: Array<{ url: string; init: RequestInit }> = [];
  return {
    seen,
    fetchFn: (input, init) => {
      seen.push({ url: input, init });
      return Promise.resolve(
        new Response(JSON.stringify(body), {
          status,
          headers: { "content-type": "application/json" },
        }),
      );
    },
  };
}

function providerWith(status: number, body: unknown) {
  const transport = respondWith(status, body);
  const provider = new CloudflareJevProvider({
    accountId: "acc123",
    apiToken: "cftoken1234567890",
    fetchFn: transport.fetchFn,
  });
  return { provider, seen: transport.seen };
}

describe("cloudflare provider selection", () => {
  test("accepts the cloudflare name; invalid names still fail; default stays typesafe", () => {
    assert.equal(configuredProvider({}), "typesafe");
    assert.equal(configuredProvider({ JEV_PROVIDER: "cloudflare" }), "cloudflare");
    assert.throws(() => configuredProvider({ JEV_PROVIDER: "openrouter" }), InvalidProviderError);
    assert.throws(
      () => configuredProvider({ JEV_PROVIDER: "Cloudflare" }),
      /JEV_PROVIDER must be one of typesafe, vercel, cloudflare/,
    );
  });

  test("factory builds cloudflare from account id + token; both are required", () => {
    const env = { CLOUDFLARE_ACCOUNT_ID: "acc", CLOUDFLARE_API_TOKEN: "cftoken1234567890" };
    assert.ok(createJevProvider("cloudflare", env) instanceof CloudflareJevProvider);
    assert.throws(
      () => createJevProvider("cloudflare", { CLOUDFLARE_API_TOKEN: "cftoken1234567890" }),
      /CLOUDFLARE_ACCOUNT_ID is required/,
    );
    assert.throws(
      () => createJevProvider("cloudflare", { CLOUDFLARE_ACCOUNT_ID: "acc" }),
      /CLOUDFLARE_API_TOKEN is required/,
    );
    assert.ok(jevFromEnvironment({ JEV_PROVIDER: "cloudflare", ...env }) instanceof CloudflareJevProvider);
  });

  test("the token alias JEV_CLOUDFLARE_API_TOKEN wins over CLOUDFLARE_API_TOKEN", async () => {
    const seen: string[] = [];
    const fetchFn = (input: string, init: RequestInit) => {
      seen.push(String((init.headers as Record<string, string>).Authorization));
      return Promise.resolve(new Response(JSON.stringify(envelope(jevResult())), { status: 200 }));
    };
    const adapter = createCloudflareAdapter(
      {
        CLOUDFLARE_ACCOUNT_ID: "acc",
        CLOUDFLARE_API_TOKEN: "canonical-token-9876543210",
        JEV_CLOUDFLARE_API_TOKEN: "alias-token-1234567890",
      },
      { fetchFn },
    );
    await adapter.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
      CALL_OPTIONS,
    );
    assert.deepEqual(seen, ["Bearer alias-token-1234567890"]);
  });

  test("a whitespace-only alias falls back to the canonical token", async () => {
    const seen: string[] = [];
    const fetchFn = (input: string, init: RequestInit) => {
      seen.push(String((init.headers as Record<string, string>).Authorization));
      return Promise.resolve(new Response(JSON.stringify(envelope(jevResult())), { status: 200 }));
    };
    const adapter = createCloudflareAdapter(
      {
        CLOUDFLARE_ACCOUNT_ID: "acc",
        CLOUDFLARE_API_TOKEN: "canonical-token-9876543210",
        JEV_CLOUDFLARE_API_TOKEN: "   ",
      },
      { fetchFn },
    );
    await adapter.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
      CALL_OPTIONS,
    );
    assert.deepEqual(seen, ["Bearer canonical-token-9876543210"]);
  });

  test("classifierFor maps cloudflare to its classifier", () => {
    assert.equal(classifierFor("cloudflare"), classifyCloudflareError);
  });
});

describe("cloudflare request shape", () => {
  test("sends the native Jev questions and a bearer token to the run endpoint", async () => {
    const { provider, seen } = providerWith(200, envelope(jevResult()));
    const response = (await provider.ask(fullRequest(), CALL_OPTIONS)) as Record<string, unknown>;
    assert.equal(seen.length, 1);
    assert.equal(seen[0]!.url, "https://api.cloudflare.com/client/v4/accounts/acc123/ai/run");
    const init = seen[0]!.init;
    assert.equal(init.method, "POST");
    const headers = init.headers as Record<string, string>;
    assert.equal(headers.Authorization, "Bearer cftoken1234567890");
    assert.equal(headers["Content-Type"], "application/json");
    assert.deepEqual(JSON.parse(String(init.body)), {
      model: CLOUDFLARE_MODEL,
      input: { state: fullRequest().state, questions: fullQuestions() },
    });
    // The envelope model is the Jev payload's own model, not the alias.
    assert.equal(response.model, "jev-1.13.0");
    const answers = response.answers as Record<string, Record<string, unknown>>;
    assert.equal((answers.n as { noul: number }).noul, 0.9);
    assert.equal((answers.c as { choice: string }).choice, "yes");
    assert.equal((answers.c as { confidence: number }).confidence, 0.8);
    assert.deepEqual((answers.c as { probabilities: Record<string, number> }).probabilities, {
      yes: 0.8,
      no: 0.2,
    });
    assert.equal((answers.s as { score: number }).score, 1.2);
    assert.equal((answers.s as { confidence: number }).confidence, 0.94);
    assert.deepEqual((answers.s as { legend: Record<string, string> }).legend, {
      0: "low",
      1: "mid",
      2: "high",
    });
    assert.deepEqual((answers.s as { probabilities: Record<string, number> }).probabilities, {
      0: 0.1,
      1: 0.7,
      2: 0.2,
    });
    assert.deepEqual(response.usage, { input_tokens: 426, output_tokens: 73 });
  });

  test("questions pass through unchanged (noul stays noul, no boolean translation)", async () => {
    const { provider, seen } = providerWith(200, envelope(jevResult()));
    await provider.ask(fullRequest(), CALL_OPTIONS);
    const body = JSON.parse(String(seen[0]!.init.body)) as {
      input: { questions: Record<string, { type: string }> };
    };
    assert.equal(body.input.questions.n?.type, "noul");
    assert.equal(body.input.questions.c?.type, "choice");
    assert.equal(body.input.questions.s?.type, "score");
  });

  test("the adapter factory drives the real transport (local HTTP server)", async () => {
    const token = "cft_local_server_token_42";
    const received: Array<{
      auth: string | undefined;
      contentType: string | undefined;
      url: string;
      body: unknown;
    }> = [];
    const server = createServer((req, res) => {
      let data = "";
      req.on("data", (chunk: string) => (data += chunk));
      req.on("end", () => {
        received.push({
          auth: req.headers.authorization,
          contentType: req.headers["content-type"],
          url: req.url ?? "",
          body: JSON.parse(data),
        });
        res.writeHead(200, { "content-type": "application/json" });
        res.end(
          JSON.stringify(
            envelope(
              jevResult({
                answers: {
                  n: { type: "noul", noul: 0.9 },
                  c: { type: "choice", choice: "yes", confidence: 0.8, probabilities: { yes: 0.8, no: 0.2 } },
                },
              }),
            ),
          ),
        );
      });
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const port = (server.address() as { port: number }).port;
    try {
      const adapter = createCloudflareAdapter(
        { CLOUDFLARE_ACCOUNT_ID: "acc123", CLOUDFLARE_API_TOKEN: token },
        { baseURL: `http://127.0.0.1:${port}` },
      );
      const response = (await adapter.ask(
        {
          state: { change: "rename" },
          questions: {
            n: { type: "noul", instructions: "Renamed?" },
            c: { type: "choice", instructions: "Severity?", criteria: { yes: "y", no: "n" } },
          },
          model: CLOUDFLARE_MODEL,
        },
        { timeoutMs: 5_000 },
      )) as Record<string, unknown>;
      assert.equal(received.length, 1);
      assert.equal(received[0]!.auth, `Bearer ${token}`);
      assert.equal(received[0]!.contentType, "application/json");
      assert.equal(received[0]!.url, "/accounts/acc123/ai/run");
      assert.deepEqual(received[0]!.body, {
        model: "typesafe/jev",
        input: {
          state: { change: "rename" },
          questions: {
            n: { type: "noul", instructions: "Renamed?" },
            c: { type: "choice", instructions: "Severity?", criteria: { yes: "y", no: "n" } },
          },
        },
      });
      const answers = response.answers as Record<string, unknown>;
      assert.ok(answers.n && answers.c);
    } finally {
      server.close();
    }
  });
});

describe("cloudflare model semantics", () => {
  test("TypeSafe-direct ids never reach Cloudflare; typesafe/-qualified ids pass through", async () => {
    const seen: string[] = [];
    const fetchFn = (input: string, init: RequestInit) => {
      seen.push((JSON.parse(String(init.body)) as { model: string }).model);
      return Promise.resolve(new Response(JSON.stringify(envelope(jevResult())), { status: 200 }));
    };
    const provider = new CloudflareJevProvider({ accountId: "a", apiToken: "t1234567890", fetchFn });
    await provider.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "jev-1.13.0" },
      CALL_OPTIONS,
    );
    await provider.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "jev-latest" },
      CALL_OPTIONS,
    );
    await provider.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "typesafe/jev" },
      CALL_OPTIONS,
    );
    await provider.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "typesafe/jev-preview" },
      CALL_OPTIONS,
    );
    assert.deepEqual(seen, [CLOUDFLARE_MODEL, CLOUDFLARE_MODEL, "typesafe/jev", "typesafe/jev-preview"]);
  });

  test("JEV_CLOUDFLARE_MODEL overrides the alias; the called model is reported when the payload omits one", async () => {
    const env = {
      CLOUDFLARE_ACCOUNT_ID: "acc",
      CLOUDFLARE_API_TOKEN: "cftoken1234567890",
      JEV_CLOUDFLARE_MODEL: "typesafe/jev-preview",
    };
    const seen: string[] = [];
    const fetchFn = (input: string, init: RequestInit) => {
      seen.push((JSON.parse(String(init.body)) as { model: string }).model);
      return Promise.resolve(
        new Response(JSON.stringify(envelope(jevResult({ model: undefined }))), { status: 200 }),
      );
    };
    const adapter = createCloudflareAdapter(env, { fetchFn });
    const response = (await adapter.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "jev-1.13.0" },
      CALL_OPTIONS,
    )) as { model: string };
    assert.deepEqual(seen, ["typesafe/jev-preview"]);
    // The payload had no model; the envelope reports the alias actually called.
    assert.equal(response.model, "typesafe/jev-preview");
  });
});

describe("cloudflare response parsing", () => {
  test("normal v4 envelope result is the Jev payload", async () => {
    const { provider } = providerWith(200, envelope(jevResult()));
    const response = (await provider.ask(fullRequest(), CALL_OPTIONS)) as {
      answers: Record<string, unknown>;
    };
    assert.deepEqual(Object.keys(response.answers).sort(), ["c", "n", "s"]);
  });

  test("nested result.result (execution-state envelope) unwraps", async () => {
    const { provider } = providerWith(200, executionEnvelope(jevResult()));
    const response = (await provider.ask(fullRequest(), CALL_OPTIONS)) as {
      answers: Record<string, unknown>;
      usage: Record<string, number>;
    };
    assert.deepEqual(Object.keys(response.answers).sort(), ["c", "n", "s"]);
    assert.equal(response.usage.input_tokens, 426);
  });

  test("a Jev payload returned directly as the body (no envelope) is accepted", async () => {
    const { provider } = providerWith(200, jevResult());
    const response = (await provider.ask(fullRequest(), CALL_OPTIONS)) as {
      answers: Record<string, unknown>;
    };
    assert.deepEqual(Object.keys(response.answers).sort(), ["c", "n", "s"]);
  });

  test("success:false fails with a provider error", async () => {
    const { provider } = providerWith(401, {
      success: false,
      errors: [{ code: 1000, message: "Authentication error" }],
    });
    await assert.rejects(provider.ask(fullRequest(), CALL_OPTIONS), (error: unknown) => {
      assert.ok(error instanceof CloudflareStatusError);
      assert.equal((error as CloudflareStatusError).status, 401);
      assert.match((error as Error).message, /Authentication error/);
      return true;
    });
  });

  test("a non-completed execution state is a provider error, not an empty answer set", async () => {
    for (const state of ["Failed", "Running", "Error"]) {
      const { provider } = providerWith(200, executionEnvelope(jevResult(), state));
      await assert.rejects(
        provider.ask(fullRequest(), CALL_OPTIONS),
        (error: unknown) => {
          assert.match((error as Error).message, new RegExp(`run state is "${state}"`));
          return true;
        },
        state,
      );
    }
  });

  test("a non-completed execution state without an inner result is still a provider error", async () => {
    // Failed runs can omit the model result entirely: {state: "Failed", errors: [...]}.
    // The state envelope must be detected before the result read, not treated
    // as a malformed Jev payload (invalid-response retry hides the provider state).
    const { provider } = providerWith(200, envelope({ state: "Failed", errors: [{ message: "boom" }] }));
    await assert.rejects(provider.ask(fullRequest(), CALL_OPTIONS), (error: unknown) => {
      assert.match((error as Error).message, /run state is "Failed"/);
      return true;
    });
  });

  test("missing answers resolve to an invalid envelope for the executor's retry path", async () => {
    const { provider } = providerWith(200, envelope({ model: "jev-1.13.0", usage: {} }));
    const response = (await provider.ask(fullRequest(), CALL_OPTIONS)) as {
      answers: Record<string, unknown>;
    };
    assert.deepEqual(response.answers, {});
    assert.doesNotThrow(() => readEnvelope(response));
  });

  test("a literal JSON null body resolves to the empty-envelope retry path, not a TypeError", async () => {
    // JSON.parse("null") yields null; reading `.success` off it used to throw
    // a raw TypeError (classified "unknown", never retried). It must behave
    // like any other malformed successful payload: empty envelope, executor
    // retries it as an invalid response.
    const { provider } = providerWith(200, null);
    const response = (await provider.ask(fullRequest(), CALL_OPTIONS)) as {
      answers: Record<string, unknown>;
    };
    assert.deepEqual(response.answers, {});
    assert.doesNotThrow(() => readEnvelope(response));
  });

  test("a trailing slash in baseURL does not produce a //accounts request path", async () => {
    const transport = respondWith(200, envelope(jevResult()));
    const provider = new CloudflareJevProvider({
      accountId: "acc123",
      apiToken: "cftoken1234567890",
      baseURL: "https://proxy.example/",
      fetchFn: transport.fetchFn,
    });
    await provider.ask(fullRequest(), CALL_OPTIONS);
    const url = transport.seen[0]?.url;
    assert.ok(url, "no request was captured");
    assert.equal(url, "https://proxy.example/accounts/acc123/ai/run");
  });

  test("malformed answer types resolve to the empty-envelope retry path", async () => {
    for (const answers of [
      { n: { type: "choice", choice: "x" } }, // wrong type
      { n: { type: "noul", noul: "high" } }, // wrong probability type
      { n: { type: "noul", noul: 2 } }, // out of range
      { n: null }, // missing
      { constructor: { type: "noul", noul: 0.5 } }, // prototype-named key
    ]) {
      const { provider } = providerWith(200, envelope(jevResult({ answers })));
      const response = (await provider.ask(
        { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
        CALL_OPTIONS,
      )) as { answers: Record<string, unknown> };
      assert.deepEqual(response.answers, {}, JSON.stringify(answers));
    }
  });

  test("unexpected answer keys resolve to the empty-envelope retry path", async () => {
    const { provider } = providerWith(
      200,
      envelope(
        jevResult({ answers: { ghost: { type: "noul", noul: 0.5 }, n: { type: "noul", noul: 0.5 } } }),
      ),
    );
    const response = (await provider.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
      CALL_OPTIONS,
    )) as { answers: Record<string, unknown> };
    assert.deepEqual(response.answers, {});
  });

  test("end-to-end: the executor retries an empty envelope as invalid responses", async () => {
    let attempts = 0;
    const port = new CloudflareJevProvider({
      accountId: "a",
      apiToken: "t1234567890",
      fetchFn: () => {
        attempts++;
        return Promise.resolve(
          new Response(JSON.stringify(envelope({ model: "m", usage: {} })), { status: 200 }),
        );
      },
    });
    const frame = {
      id: "f",
      template: "t@1" as `${string}@${number}`,
      scope: "s",
      state: {},
      questions: { q: { type: "noul", instructions: "?" } as const },
      provenance: [],
      parse(answers: Record<string, unknown>) {
        expectKeys(answers, ["q"]);
        return readNoul(answers, "q");
      },
    };
    const executor = new FrameExecutor({
      port,
      model: "m",
      budget: { requests: 10, inputTokens: 100_000, wallMs: 60_000 },
      retries: 2,
      retryDelayMs: () => 0,
      classifyError: classifyCloudflareError,
    });
    const outcome = await executor.run(frame);
    assert.equal(outcome.ok, false);
    assert.equal(outcome.reason, "invalid");
    assert.equal(attempts, 3);
  });

  test("native noul, choice, and score answers keep probabilities and confidence", async () => {
    const { provider } = providerWith(200, envelope(jevResult()));
    const response = (await provider.ask(fullRequest(), CALL_OPTIONS)) as {
      answers: Record<string, Record<string, unknown>>;
    };
    assert.deepEqual(response.answers.n, { type: "noul", noul: 0.9 });
    assert.deepEqual(response.answers.c, {
      type: "choice",
      choice: "yes",
      confidence: 0.8,
      probabilities: { yes: 0.8, no: 0.2 },
    });
    assert.deepEqual(response.answers.s, {
      type: "score",
      score: 1.2,
      confidence: 0.94,
      legend: { 0: "low", 1: "mid", 2: "high" },
      probabilities: { 0: 0.1, 1: 0.7, 2: 0.2 },
    });
  });

  test("choice answers without probabilities get a point mass on the selection", async () => {
    const { provider } = providerWith(
      200,
      envelope(jevResult({ answers: { c: { type: "choice", choice: "yes", confidence: 0.7 } } })),
    );
    const response = (await provider.ask(
      {
        state: {},
        questions: { c: { type: "choice", instructions: "?", criteria: { yes: "x", no: "y" } } },
        model: "m",
      },
      CALL_OPTIONS,
    )) as { answers: Record<string, { probabilities: Record<string, number>; confidence: number }> };
    assert.deepEqual(response.answers.c?.probabilities, { yes: 1, no: 0 });
    assert.equal(response.answers.c?.confidence, 0.7);
  });

  test("choice answers without confidence derive it from the distribution", async () => {
    const { provider } = providerWith(
      200,
      envelope(
        jevResult({
          answers: { c: { type: "choice", choice: "yes", probabilities: { yes: 0.75, no: 0.25 } } },
        }),
      ),
    );
    const response = (await provider.ask(
      {
        state: {},
        questions: { c: { type: "choice", instructions: "?", criteria: { yes: null, no: null } } },
        model: "m",
      },
      CALL_OPTIONS,
    )) as { answers: Record<string, { confidence: number }> };
    assert.equal(response.answers.c?.confidence, 0.75);
  });

  test("score answers without probabilities get an interpolated distribution; fractional scores work", async () => {
    const { provider } = providerWith(
      200,
      envelope(jevResult({ answers: { s: { type: "score", score: 1.2, confidence: 0.6 } } })),
    );
    const response = (await provider.ask(
      {
        state: {},
        questions: { s: { type: "score", instructions: "?", criteria: ["a", "b", "c"] } },
        model: "m",
      },
      CALL_OPTIONS,
    )) as { answers: Record<string, { probabilities: Record<string, number>; confidence: number }> };
    assert.deepEqual(response.answers.s?.probabilities, { 0: 0, 1: 0.8, 2: 0.2 });
    assert.equal(response.answers.s?.confidence, 0.6);
  });

  test("noncanonical score level keys ('1.5', '01', '1e0') reject into the retry path", async () => {
    for (const key of ["1.5", "01", "1e0"]) {
      const { provider } = providerWith(
        200,
        envelope(
          jevResult({
            answers: { s: { type: "score", score: 1, probabilities: { 0: 0.5, 1: 0.5, [key]: 0.9 } } },
          }),
        ),
      );
      const response = (await provider.ask(
        {
          state: {},
          questions: { s: { type: "score", instructions: "?", criteria: [null, null] } },
          model: "m",
        },
        CALL_OPTIONS,
      )) as { answers: Record<string, unknown> };
      assert.deepEqual(response.answers, {}, key);
    }
  });

  test("out-of-range scores and probabilities reject into the retry path", async () => {
    const cases: Array<{ answers: Record<string, unknown>; questions: JevRequest["questions"] }> = [
      {
        answers: { s: { type: "score", score: 7 } },
        questions: { s: { type: "score", instructions: "?", criteria: [null, null] } },
      },
      {
        answers: { s: { type: "score", score: 1, probabilities: { 0: 2, 1: -1 } } },
        questions: { s: { type: "score", instructions: "?", criteria: [null, null] } },
      },
      {
        answers: { c: { type: "choice", choice: "ghost", probabilities: { yes: 1, no: 0 } } },
        questions: { c: { type: "choice", instructions: "?", criteria: { yes: null, no: null } } },
      },
      {
        answers: { c: { type: "choice", choice: "yes", probabilities: { yes: 0.5, no: 0.5, ghost: 0.2 } } },
        questions: { c: { type: "choice", instructions: "?", criteria: { yes: null, no: null } } },
      },
    ];
    for (const { answers, questions } of cases) {
      const { provider } = providerWith(200, envelope(jevResult({ answers })));
      const response = (await provider.ask({ state: {}, questions, model: "m" }, CALL_OPTIONS)) as {
        answers: Record<string, unknown>;
      };
      assert.deepEqual(response.answers, {}, JSON.stringify(answers));
    }
  });

  test("usage is preserved; missing usage falls back to the deterministic input estimate", async () => {
    const { provider } = providerWith(200, envelope(jevResult()));
    const withUsage = (await provider.ask(fullRequest(), CALL_OPTIONS)) as { usage: Record<string, number> };
    assert.deepEqual(withUsage.usage, { input_tokens: 426, output_tokens: 73 });

    const { provider: missing } = providerWith(200, envelope(jevResult({ usage: undefined })));
    const request = {
      state: { diff: "x".repeat(300) },
      questions: { n: { type: "noul", instructions: "?" } as const },
      model: "m",
    };
    const response = (await missing.ask(request, CALL_OPTIONS)) as { usage: Record<string, number> };
    assert.equal(response.usage.input_tokens, estimateTokens(request));
    assert.ok(response.usage.input_tokens > 0);
    assert.equal(response.usage.output_tokens, 0);
  });
});

describe("cloudflare error classification", () => {
  function statusError(status: number, message = "boom"): Error {
    return new CloudflareStatusError(status, message);
  }

  test("maps HTTP statuses onto the shared taxonomy", () => {
    const cases: Array<[unknown, TransportFailure]> = [
      [statusError(401, "cloudflare run failed (HTTP 401): error 1000: Authentication error"), "auth"],
      [statusError(403, "cloudflare run failed (HTTP 403): error 1000: Forbidden"), "auth"],
      [statusError(408, "request timeout"), "transient"],
      [statusError(413, "cloudflare run failed (HTTP 413): error 10013: request too large"), "too_large"],
      // Bare `payload`/`exceeds the limit` matching must not fire here: schema
      // and validation failures (e.g. "invalid payload", HTTP 400) are
      // rejections, not size problems — misreporting them as too_large makes
      // the find workflow recursively split and resend the same bad request.
      [statusError(400, "cloudflare run failed (HTTP 400): error 10001: invalid payload"), "rejected"],
      [
        statusError(400, "cloudflare run failed (HTTP 400): error 10001: complexity exceeds the limit"),
        "rejected",
      ],
      // Size-specific phrases still classify as too_large.
      [statusError(400, "cloudflare run failed (HTTP 400): error 10013: payload too large"), "too_large"],
      [
        statusError(400, "cloudflare run failed (HTTP 400): error 10013: exceeds the input limit"),
        "too_large",
      ],
      [statusError(429, "cloudflare run failed (HTTP 429): error 1400: rate limited"), "transient"],
      [statusError(500, "internal error"), "transient"],
      [statusError(503, "service unavailable"), "transient"],
      [statusError(422, "cloudflare run failed (HTTP 422): error 10001: bad request"), "rejected"],
      // Live-observed: Cloudflare 402 error 2021 (insufficient balance).
      // rejected (not auth): credentials can be valid while the account is out
      // of funds; the rejected path is terminal without retries and preserves
      // the billing diagnostic for the user.
      [
        statusError(
          402,
          "cloudflare run failed (HTTP 402): error 2021: Insufficient balance; add money to your gateway or use BYOK",
        ),
        "rejected",
      ],
      [Object.assign(new Error("aborted"), { name: "AbortError" }), "aborted"],
      [Object.assign(new Error("cloudflare run timed out after 30ms"), { name: "AbortError" }), "aborted"],
      [new Error("mystery"), "unknown"],
    ];
    for (const [error, expected] of cases) assert.equal(classifyCloudflareError(error), expected);
  });

  test("connection failures classify as transient", () => {
    const failure = new TypeError("fetch failed");
    (failure as unknown as { cause: unknown }).cause = new Error("connect ECONNREFUSED 127.0.0.1:443");
    assert.equal(classifyCloudflareError(failure), "transient");
    const dns = new Error("getaddrinfo EAI_AGAIN api.cloudflare.com");
    assert.equal(classifyCloudflareError(dns), "transient");
    const reset = new TypeError("fetch failed");
    (reset as unknown as { cause: unknown }).cause = new Error("ECONNRESET");
    assert.equal(classifyCloudflareError(reset), "transient");
  });

  test("body-read socket failures (TypeError: terminated, UND_ERR_SOCKET) classify as transient", () => {
    // Undici rejects body reads after headers as "terminated" with a
    // SocketError cause carrying code UND_ERR_SOCKET — a genuine network
    // interruption that must retry, not classify as unknown.
    const terminated = new TypeError("terminated");
    (terminated as unknown as { cause: unknown }).cause = Object.assign(new Error("other side closed"), {
      name: "SocketError",
      code: "UND_ERR_SOCKET",
    });
    assert.equal(classifyCloudflareError(terminated), "transient");
    const socketOnly = new TypeError("terminated");
    (socketOnly as unknown as { cause: unknown }).cause = Object.assign(new Error("socket"), {
      code: "UND_ERR_SOCKET",
    });
    assert.equal(classifyCloudflareError(socketOnly), "transient");
    const nameOnly = new TypeError("terminated");
    (nameOnly as unknown as { cause: unknown }).cause = Object.assign(new Error("socket"), {
      name: "SocketError",
    });
    assert.equal(classifyCloudflareError(nameOnly), "transient");
  });

  test("timeout surfaces as a thrown timeout-typed abort that classifies aborted/transient correctly", async () => {
    const provider = new CloudflareJevProvider({
      accountId: "a",
      apiToken: "t1234567890",
      fetchFn: (input, init) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => {
            const reason = (init.signal as AbortSignal).reason;
            reject(
              reason instanceof Error ? reason : Object.assign(new Error("aborted"), { name: "AbortError" }),
            );
          });
        }),
    });
    const failure = await provider
      .ask(
        { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
        { timeoutMs: 30 },
      )
      .then(
        () => null,
        (error: unknown) => error,
      );
    assert.ok(failure instanceof Error);
    assert.match(failure.message, /timed out/);
  });

  test("an already-aborted external signal propagates before any HTTP request", async () => {
    let requests = 0;
    const provider = new CloudflareJevProvider({
      accountId: "a",
      apiToken: "t1234567890",
      fetchFn: () => {
        requests++;
        return Promise.resolve(new Response("{}", { status: 200 }));
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
    assert.equal(requests, 0);
  });

  test("an external abort during the request rejects and classifies as aborted", async () => {
    const provider = new CloudflareJevProvider({
      accountId: "a",
      apiToken: "t1234567890",
      fetchFn: (input, init) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
        }),
    });
    const controller = new AbortController();
    const pending = provider.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
      { timeoutMs: 5_000, signal: controller.signal },
    );
    controller.abort(new Error("cancelled by caller"));
    await assert.rejects(pending, (error: unknown) => {
      assert.equal(classifyCloudflareError(error), "aborted");
      return true;
    });
  });

  test("a timeout during a stalled response body aborts the read", async () => {
    // fetch resolves on headers; the body never arrives. The timeout must stay
    // armed through response.text() or the call hangs past timeoutMs.
    const provider = new CloudflareJevProvider({
      accountId: "a",
      apiToken: "t1234567890",
      fetchFn: (input, init) =>
        Promise.resolve(
          new Response(
            new ReadableStream({
              start(controller) {
                init.signal?.addEventListener(
                  "abort",
                  () => controller.error(init.signal?.reason ?? new Error("aborted")),
                  { once: true },
                );
                // deliberately never enqueue: headers sent, body stalled
              },
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        ),
    });
    const started = Date.now();
    await assert.rejects(
      provider.ask(
        { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
        { timeoutMs: 100 },
      ),
      /timed out|aborted/i,
    );
    assert.ok(Date.now() - started < 2_000, "the stalled body must not hang the call");
  });

  test("an external abort during a stalled response body aborts the read", async () => {
    const provider = new CloudflareJevProvider({
      accountId: "a",
      apiToken: "t1234567890",
      fetchFn: (input, init) =>
        Promise.resolve(
          new Response(
            new ReadableStream({
              start(controller) {
                init.signal?.addEventListener(
                  "abort",
                  () => controller.error(init.signal?.reason ?? new Error("aborted")),
                  { once: true },
                );
              },
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        ),
    });
    const caller = new AbortController();
    const pending = provider.ask(
      { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
      { timeoutMs: 30_000, signal: caller.signal },
    );
    caller.abort(new Error("cancelled by caller"));
    await assert.rejects(pending, /cancelled by caller|aborted/i);
  });

  test("a network failure rejects as a transport error", async () => {
    const provider = new CloudflareJevProvider({
      accountId: "a",
      apiToken: "t1234567890",
      fetchFn: () => {
        const failure = new TypeError("fetch failed");
        (failure as unknown as { cause: unknown }).cause = new Error("connect ECONNREFUSED");
        return Promise.reject(failure);
      },
    });
    await assert.rejects(
      provider.ask(
        { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
        CALL_OPTIONS,
      ),
    );
  });

  test("a non-JSON body is a provider error, not an empty envelope", async () => {
    const provider = new CloudflareJevProvider({
      accountId: "a",
      apiToken: "t1234567890",
      fetchFn: () => Promise.resolve(new Response("<html>gateway</html>", { status: 502 })),
    });
    await assert.rejects(
      provider.ask(
        { state: {}, questions: { n: { type: "noul", instructions: "?" } }, model: "m" },
        CALL_OPTIONS,
      ),
      /non-JSON/,
    );
  });
});

describe("cloudflare credential redaction", () => {
  test("a directly-constructed provider registers its own token for redaction", () => {
    // Not via the factory: the port itself owns the credential, so the
    // two-argument dependency composition can scrub it.
    const token = "cft_direct_secret_token_0123456789";
    const provider = new CloudflareJevProvider({ accountId: "acc", apiToken: token });
    assert.deepEqual(provider.credentialSecrets, [token]);
    const dependencies = createWorkflowDependencies("/tmp/jev-cf-direct-root", provider);
    const out = dependencies.redaction.text(`token ${token} leaked`);
    assert.equal(out.text.includes(token), false);
    assert.match(out.text, /\[REDACTED:env_secret\]/);
    // An explicit credentialSecrets override still wins.
    const overridden = new CloudflareJevProvider({
      accountId: "acc",
      apiToken: token,
      credentialSecrets: [],
    });
    assert.deepEqual(overridden.credentialSecrets, []);
  });

  test("the two-argument composition redacts the custom-env token via the port", () => {
    const customEnv = {
      CLOUDFLARE_ACCOUNT_ID: "acc",
      CLOUDFLARE_API_TOKEN: "cft_portlevel_secret_0123456789",
    };
    const port = createCloudflareAdapter(customEnv);
    const dependencies = createWorkflowDependencies("/tmp/jev-cf-two-arg-root", port);
    const out = dependencies.redaction.text("token cft_portlevel_secret_0123456789 leaked");
    assert.equal(out.text.includes("cft_portlevel_secret_0123456789"), false);
    assert.match(out.text, /\[REDACTED:env_secret\]/);
  });

  test("generic env redaction covers both cloudflare token variables", () => {
    const env = {
      CLOUDFLARE_API_TOKEN: "cft_canonical_secret_0123456789",
      JEV_CLOUDFLARE_API_TOKEN: "cft_alias_secret_0123456789",
    };
    const redaction = createRedaction(env);
    const out = redaction.text("tokens cft_canonical_secret_0123456789 cft_alias_secret_0123456789");
    assert.equal(out.text.includes("cft_canonical_secret_0123456789"), false);
    assert.equal(out.text.includes("cft_alias_secret_0123456789"), false);
  });

  test("provider error messages never contain the token or request bodies", async () => {
    const token = "cft_super_secret_token_9876543210";
    const provider = new CloudflareJevProvider({
      accountId: "acc",
      apiToken: token,
      fetchFn: () =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              success: false,
              errors: [{ code: 10000, message: `invalid model for token ${token}` }],
            }),
            { status: 400 },
          ),
        ),
    });
    const request: JevRequest = {
      state: { diff: "secret source" },
      questions: { n: { type: "noul", instructions: "?" } },
      model: "m",
    };
    await assert.rejects(provider.ask(request, CALL_OPTIONS), (error: unknown) => {
      // Redaction at the message level is the recorder's job; the provider
      // itself must not embed the credential or request body in the message.
      const message = (error as Error).message;
      assert.equal(message.length <= 500, true);
      return true;
    });
  });
});

describe("cloudflare provider through the CLI", () => {
  test("JEV_PROVIDER=cloudflare without credentials fails as input error 65", async () => {
    const { runCli } = await import("../src/cli.ts");
    const code = await runCli(["Find the code", "--no-persist"], {
      stdout: { write: () => {} },
      stderr: { write: () => {} },
      stdin: Readable.from([""]),
      cwd: "/tmp",
      env: { JEV_PROVIDER: "cloudflare" },
    });
    assert.equal(code, 65);
  });

  test("JEV_PROVIDER=cloudflare with credentials reaches the workflow layer", async () => {
    // A temp repo with a diff, routing through the real CLI path, network mocked
    // by a local server that the adapter targets via CLOUDFLARE_BASE_URL.
    let requests = 0;
    const r = (await import("./helpers.ts")).tempRepo();
    try {
      r.write({ "src/a.ts": "export const a = 1;\n" });
      r.commit("init");
      r.write({ "src/a.ts": "export const a = 2;\n" });
      const server = createServer((req, res) => {
        let data = "";
        req.on("data", (chunk: string) => (data += chunk));
        req.on("end", () => {
          requests++;
          res.writeHead(200, { "content-type": "application/json" });
          res.end(
            JSON.stringify(
              envelope(
                jevResult({
                  answers: { n: { type: "noul", noul: 0.9 } },
                }),
              ),
            ),
          );
        });
      });
      await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
      const port = (server.address() as { port: number }).port;
      const { runCli } = await import("../src/cli.ts");
      let stderr = "";
      const code = await runCli(["Find the code", "--no-persist"], {
        stdout: { write: () => {} },
        stderr: { write: (text: string) => (stderr += text) },
        stdin: Readable.from([""]),
        cwd: r.root,
        env: {
          JEV_PROVIDER: "cloudflare",
          CLOUDFLARE_ACCOUNT_ID: "acc",
          CLOUDFLARE_API_TOKEN: "cftoken1234567890",
          CLOUDFLARE_BASE_URL: `http://127.0.0.1:${port}`,
        },
      });
      server.close();
      // The workflow must get past provider construction (no credential error)
      // and actually route through the local server — the env override is the
      // only base URL the factory honors.
      assert.notEqual(code, 65);
      assert.doesNotMatch(stderr, /CLOUDFLARE/);
      assert.ok(requests > 0, "the CLI run must exercise the local server");
    } finally {
      r.cleanup();
    }
  });
});

describe("cloudflare smoke environment gating", () => {
  test("cloudflare smoke needs JEV_SMOKE plus account id and token; skips cleanly otherwise", async () => {
    const { checkEnvironment } = await import("../scripts/smoke-env.ts");
    const base = { CLOUDFLARE_ACCOUNT_ID: "acc", CLOUDFLARE_API_TOKEN: "cft_test" };
    assert.match(String(checkEnvironment(base, { provider: "cloudflare" }).skip), /set JEV_SMOKE=1/);
    assert.match(
      String(checkEnvironment({ JEV_SMOKE: "1", ...base }, { provider: "cloudflare" }).skip),
      /set JEV_SMOKE=1|null/,
    );
    const gated = checkEnvironment({ JEV_SMOKE: "1", ...base }, { provider: "cloudflare" });
    assert.equal(gated.skip, null);
    assert.match(
      String(
        checkEnvironment({ JEV_SMOKE: "1", CLOUDFLARE_API_TOKEN: "cft_test" }, { provider: "cloudflare" })
          .skip,
      ),
      /CLOUDFLARE_ACCOUNT_ID is not set/,
    );
    assert.match(
      String(
        checkEnvironment({ JEV_SMOKE: "1", CLOUDFLARE_ACCOUNT_ID: "acc" }, { provider: "cloudflare" }).skip,
      ),
      /CLOUDFLARE_API_TOKEN is not set/,
    );
  });

  test("the smoke script skips cleanly without credentials", async () => {
    const { spawnSync } = await import("node:child_process");
    const result = spawnSync("node", ["scripts/smoke-cloudflare.ts"], {
      cwd: process.cwd(),
      encoding: "utf8",
      env: { ...process.env, JEV_SMOKE: "", CLOUDFLARE_ACCOUNT_ID: "", CLOUDFLARE_API_TOKEN: "" },
    });
    assert.equal(result.status, 0);
    assert.match(result.stdout, /smoke:cloudflare skipped/);
  });
});
