#!/usr/bin/env node
// Runs only help, version, and deliberately invalid invocations, so it is safe
// against a CLI that follows the checklist; one that acts on them may run.
// Each run has no TTY on stdin, NO_COLOR=1, and a 10s limit.
// With a usage spec, invalid invocations it rejects also run against commands
// declared effect="read", because a CLI that wrongly accepts one runs it.

import { spawnSync } from "node:child_process";

const [bin, specFile] = process.argv.slice(2);
if (bin === undefined) {
  console.error("usage: probe.mjs <bin> [app.usage.kdl]");
  process.exit(2);
}

const env = { ...process.env, NO_COLOR: "1" };
delete env.CLICOLOR_FORCE;

function run(args) {
  const start = Date.now();
  const result = spawnSync(bin, args, {
    input: "",
    encoding: "utf8",
    timeout: 10_000,
    env,
  });
  return {
    ms: Date.now() - start,
    code: result.status,
    timedOut: result.error?.code === "ETIMEDOUT",
    out: result.stdout ?? "",
    err: result.stderr || (result.error?.message ?? ""),
  };
}

const help = run(["--help"]);
const version = run(["--version"]);
if (help.code !== 0 && version.code === help.code && version.err === help.err) {
  console.log(
    `BROKEN ${bin}: --help and --version both fail the same way; check that it is installed and runs:`,
  );
  console.log(help.err.split("\n").slice(0, 5).join("\n"));
  process.exit(2);
}

let fails = 0;

function expect(want, stream, args, { optional = false, sameAs } = {}) {
  const result = run(args);
  const problems = [];
  if (result.timedOut) {
    problems.push("no exit within 10s without a TTY (prompt or hang?)");
  } else {
    if (result.code !== want) {
      problems.push(`exit ${result.code}, want ${want}`);
    }
    if (result[stream] === "") problems.push(`empty std${stream}`);
    if (stream === "err" && result.out.trim() !== "") {
      problems.push("error output on stdout");
    }
    if ((result.out + result.err).includes("\x1b")) {
      problems.push("ANSI escapes despite NO_COLOR and no TTY");
    }
    if (sameAs !== undefined && result.out !== sameAs.result.out) {
      problems.push(`stdout differs from ${sameAs.label}`);
    }
  }
  const label = [bin, ...args].join(" ");
  if (problems.length === 0) {
    console.log(`PASS ${label}`);
  } else if (optional) {
    console.log(
      `NOTE ${label}: ${
        problems.join("; ")
      } (fine if the framework does not support this form)`,
    );
  } else {
    console.log(`FAIL ${label}: ${problems.join("; ")}`);
    fails++;
  }
  return result;
}

expect(0, "out", []);
const rootHelp = { label: "--help", result: expect(0, "out", ["--help"]) };
const rootVersion = {
  label: "--version",
  result: expect(0, "out", ["--version"]),
};
if (rootVersion.result.ms > 500) {
  console.log(
    `NOTE ${bin} --version: took ${rootVersion.result.ms}ms, over 500ms`,
  );
}
if (specFile === undefined) {
  console.log(`NOTE ${bin}: no spec, so subcommands are not probed`);
}
expect(0, "out", ["-h"]);
expect(0, "out", ["help"], { optional: true, sameAs: rootHelp });
expect(0, "out", ["-V"], { optional: true, sameAs: rootVersion });
expect(0, "out", ["version"], { optional: true, sameAs: rootVersion });
expect(0, "out", ["-v"], { sameAs: rootVersion });
expect(2, "err", ["__probe_unknown_command__"]);
expect(2, "err", ["--__probe-unknown-flag__"]);

function specCommands() {
  const json = spawnSync("usage", ["generate", "json", "-f", specFile], {
    encoding: "utf8",
  });
  if (json.status !== 0) {
    console.error(json.stderr || json.error?.message);
    process.exit(2);
  }
  const commands = [];
  const walk = (cmd, path) => {
    for (const [name, sub] of Object.entries(cmd.subcommands ?? {})) {
      if (sub.hide) continue;
      commands.push({ path: [...path, name], cmd: sub });
      walk(sub, [...path, name]);
    }
  };
  walk(JSON.parse(json.stdout).cmd, []);
  return commands;
}

function spelling(flag) {
  return flag.long.length > 0 ? `--${flag.long[0]}` : `-${flag.short[0]}`;
}

function specRejects(args) {
  const explain = spawnSync(
    "usage",
    ["explain", "-f", specFile, "--format", "json", "--", bin, ...args],
    { encoding: "utf8" },
  );
  if (explain.status !== 0) {
    console.error(explain.stderr || explain.error?.message);
    process.exit(2);
  }
  const report = JSON.parse(explain.stdout);
  return report.errors.length > 0 || report.refused !== null;
}

for (const { path, cmd } of specFile === undefined ? [] : specCommands()) {
  const label = path.join(" ");
  const subHelp = {
    label: `${label} --help`,
    result: expect(0, "out", [...path, "--help"]),
  };
  expect(0, "out", [...path, "-h"]);
  expect(0, "out", ["help", ...path], { optional: true, sameAs: subHelp });

  for (const flag of cmd.flags.filter((f) => !f.hide && f.long.length > 0)) {
    if (!new RegExp(`--${flag.long[0]}(?![\\w-])`).test(subHelp.result.out)) {
      console.log(
        `FAIL ${bin} ${label}: spec flag --${
          flag.long[0]
        } is missing from --help`,
      );
      fails++;
    }
  }
  if (cmd.effect !== "read") {
    console.log(
      `NOTE ${bin} ${label}: effect is ${
        cmd.effect ?? "unknown"
      }, so rejected invocations are not run`,
    );
    continue;
  }
  const filled = [
    ...path,
    ...cmd.flags.filter((f) => f.required).flatMap((f) => [
      spelling(f),
      ...(f.arg ? [f.arg.choices?.choices[0] ?? "x"] : []),
    ]),
    ...cmd.args.filter((a) => a.required).map(() => "x"),
  ];
  const cases = [
    path,
    [...filled, "--__probe-unknown-flag__"],
    [...filled, "__probe_extra__"],
    ...cmd.flags.filter((f) => f.arg?.choices).map((
      f,
    ) => [...filled, spelling(f), "__probe_bad_choice__"]),
  ];
  for (const args of cases.filter(specRejects)) expect(2, "err", args);
}

console.log(`${fails} failure(s)`);
process.exitCode = fails === 0 ? 0 : 1;
