# Background Process

One long-lived process owns the state; CLI, TUI, web, and MCP front ends are thin clients over a local socket of mode 0600 in a directory of mode 0700.

- `start` and `stop` are idempotent.
- Clients auto-start the daemon unless a setting disables it: re-execute the current binary, plus the script path under an interpreter, detached, with output appended to a log in the state directory; if the child exits before accepting connections, show that log as the error.
- The first message on each connection carries the protocol version, app version, and pid; on a mismatch, the client fails and names the stop command.
- The daemon holds an `flock` on a lock file that records its pid. Stopping sends SIGTERM to that pid only while the lock is held, then waits for its release, so it works across protocol versions and never signals a reused pid.
- Register the binary by absolute path, without `~`, `PATH` lookup, or a shell, and write the registering shell's `PATH` into the unit.
- Restart only after a crash: launchd `KeepAlive = { SuccessfulExit = false }`, systemd `Restart=on-failure`.
- A Unix socket path fits in 103 bytes on macOS and 107 on Linux, leaving room for the terminating NUL; check at startup.
