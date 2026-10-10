package cli

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/h3y6e/anna/internal/core"
)

func TestWhenVersionIsRequestedWithVItMatchesTheLongVersionFlag(t *testing.T) {
	t.Parallel()

	// Act
	long, longErrOut, longErr := executeCommand("--version")
	short, shortErrOut, shortErr := executeCommand("-v")

	// Assert
	if longErr != nil || shortErr != nil {
		t.Fatalf("version errors = %v %v\nstderr: %s %s", longErr, shortErr, longErrOut, shortErrOut)
	}
	if long != short || !strings.Contains(long, "dev") {
		t.Fatalf("version output = %q and %q, want the same version text", long, short)
	}
}

func TestWhenASubcommandReceivesVItIsAUsageError(t *testing.T) {
	t.Parallel()

	// Act
	stdout, _, err := executeCommand("recall", "-v")

	// Assert
	if ExitCode(err) != 2 {
		t.Fatalf("exit = %d, error = %v, want usage exit 2", ExitCode(err), err)
	}
	if strings.TrimSpace(stdout) != "" {
		t.Fatalf("stdout = %q, want empty", stdout)
	}
	if !strings.Contains(err.Error(), "--help") {
		t.Fatalf("error = %q, want a pointer to --help", err)
	}
}

func TestWhenTheCommandIsUnknownTheErrorSuggestsACommandAndExitsAsUsage(t *testing.T) {
	t.Parallel()

	// Act
	_, _, err := executeCommand("recoll")

	// Assert
	if ExitCode(err) != 2 {
		t.Fatalf("exit = %d, error = %v, want usage exit 2", ExitCode(err), err)
	}
	if !strings.Contains(err.Error(), "Did you mean") || !strings.Contains(err.Error(), "recall") {
		t.Fatalf("error = %q, want a recall suggestion", err)
	}
	if !strings.Contains(err.Error(), "--help") {
		t.Fatalf("error = %q, want a pointer to --help", err)
	}
}

func TestWhenHelpIsRequestedItDocumentsTheUsageExitCode(t *testing.T) {
	t.Parallel()

	// Act
	stdout, stderr, err := executeCommand("--help")

	// Assert
	if err != nil {
		t.Fatalf("help failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "usage error") {
		t.Fatalf("help = %q, want the usage exit code", stdout)
	}
}

func TestWhenJSONIsSetAUsageErrorIsAResultRecordOnStdout(t *testing.T) {
	t.Parallel()

	// Act
	stdout, _, err := executeCommand("--json", "--nope")

	// Assert
	if ExitCode(err) != 2 {
		t.Fatalf("exit = %d, error = %v, want usage exit 2", ExitCode(err), err)
	}
	if !strings.Contains(stdout, `"record":"result"`) || !strings.Contains(stdout, `"code":"usage"`) || !strings.Contains(stdout, `"ok":false`) {
		t.Fatalf("stdout = %q, want a usage result record", stdout)
	}
}

func TestWhenJSONIsGivenNremEndsWithAResultRecord(t *testing.T) {
	t.Parallel()

	// Arrange
	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nJSON output ends with a result record.\n")

	// Act
	stdout, stderr, err := executeCommand("nrem", source, "--json")

	// Assert
	if err != nil {
		t.Fatalf("nrem failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, `"record":"index"`) || !strings.Contains(stdout, `"record":"result"`) || !strings.Contains(stdout, `"ok":true`) {
		t.Fatalf("stdout = %q, want an index line and a result record", stdout)
	}
}

func TestWhenDryRunIsGivenNremWritesNothing(t *testing.T) {
	t.Parallel()

	// Arrange
	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nDry run should not create a memory.\n")
	var embedded bool

	// Act
	stdout, stderr, err := executeCommandWithDependencies(testDependencies(Dependencies{
		TextSource: defaultTestDependencies().TextSource,
		IndexStore: defaultTestDependencies().IndexStore,
		NewEmbedder: func(EmbedderSettings) (core.Embedder, error) {
			embedded = true
			return fakeEmbedder{}, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			return fakeTokenizer{}, nil
		},
	}), "nrem", source, "--dry-run")

	// Assert
	if err != nil {
		t.Fatalf("nrem dry-run failed: %v\nstderr: %s", err, stderr)
	}
	if embedded {
		t.Fatal("dry-run called the embedder")
	}
	if _, statErr := os.Stat(filepath.Join(source, ".anna.db")); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("memory stat error = %v, want not exist", statErr)
	}
	if !strings.Contains(stdout, "would consolidate 1 documents") {
		t.Fatalf("stdout = %q, want the documents that would be written", stdout)
	}
}

func TestWhenAmnesiaWouldOverwriteWithoutYesNremRefuses(t *testing.T) {
	t.Parallel()

	// Arrange
	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nAn existing memory must not be overwritten without confirmation.\n")
	if _, _, err := executeCommand("nrem", source); err != nil {
		t.Fatalf("seed nrem failed: %v", err)
	}
	var embedded bool

	// Act
	_, stderr, err := executeCommandWithDependencies(testDependencies(Dependencies{
		TextSource: defaultTestDependencies().TextSource,
		IndexStore: defaultTestDependencies().IndexStore,
		NewEmbedder: func(EmbedderSettings) (core.Embedder, error) {
			embedded = true
			return fakeEmbedder{}, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			return fakeTokenizer{}, nil
		},
	}), "nrem", source, "--amnesia")

	// Assert
	if ExitCode(err) != 2 {
		t.Fatalf("exit = %d, error = %v, want usage exit 2", ExitCode(err), err)
	}
	if embedded {
		t.Fatal("refused amnesia still called the embedder")
	}
	if !strings.Contains(stderr, "overwrite\t"+filepath.Join(source, ".anna.db")) {
		t.Fatalf("stderr = %q, want the memory that would be overwritten", stderr)
	}
	if !strings.Contains(err.Error(), "--yes") {
		t.Fatalf("error = %q, want --yes", err)
	}
}

func TestWhenAmnesiaIsGivenWithYesNremRebuildsTheMemory(t *testing.T) {
	t.Parallel()

	// Arrange
	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nConfirmed amnesia rebuilds the memory.\n")
	if _, _, err := executeCommand("nrem", source); err != nil {
		t.Fatalf("seed nrem failed: %v", err)
	}

	// Act
	stdout, stderr, err := executeCommand("nrem", source, "--amnesia", "--yes")

	// Assert
	if err != nil {
		t.Fatalf("nrem --amnesia --yes failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "consolidated 1 documents") {
		t.Fatalf("stdout = %q, want a rebuilt memory", stdout)
	}
}

func TestWhenSettingsRunsItShowsTheDefaultMemoryOrigin(t *testing.T) {
	t.Parallel()

	// Act
	stdout, stderr, err := executeCommand("settings")

	// Assert
	if err != nil {
		t.Fatalf("settings failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "memory\t.anna.db\tdefault") {
		t.Fatalf("stdout = %q, want the default memory origin", stdout)
	}
}

func TestWhenTheMemoryFlagIsGivenSettingsReportsTheFlagOrigin(t *testing.T) {
	t.Parallel()

	// Act
	stdout, stderr, err := executeCommand("settings", "--memory", "flag.db")

	// Assert
	if err != nil {
		t.Fatalf("settings failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "memory\tflag.db\tflag") {
		t.Fatalf("stdout = %q, want a flag origin", stdout)
	}
}

func TestWhenTheMemoryEnvironmentVariableIsSetSettingsReportsTheEnvOrigin(t *testing.T) {
	// Arrange
	t.Setenv("ANNA_MEMORY", "env.db")

	// Act
	stdout, stderr, err := executeCommand("settings")

	// Assert
	if err != nil {
		t.Fatalf("settings failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "memory\tenv.db\tenv") {
		t.Fatalf("stdout = %q, want an env origin", stdout)
	}
}

func TestWhenSettingsJSONIsRequestedTheAPIKeyIsRedacted(t *testing.T) {
	// Arrange
	t.Setenv("ANNA_EMBEDDER_API_KEY", "super-secret")

	// Act
	stdout, stderr, err := executeCommand("settings", "--json")

	// Assert
	if err != nil {
		t.Fatalf("settings failed: %v\nstderr: %s", err, stderr)
	}
	if strings.Contains(stdout, "super-secret") {
		t.Fatalf("stdout = %q, want the API key redacted", stdout)
	}
	if !strings.Contains(stdout, `"redacted":true`) || !strings.Contains(stdout, `"record":"result"`) {
		t.Fatalf("stdout = %q, want a redacted setting and a result record", stdout)
	}
}

func TestWhenALocalConfigExistsItOverridesTheProjectConfig(t *testing.T) {
	// Arrange
	dir := tempDir(t)
	t.Chdir(dir)
	writeFile(t, filepath.Join(dir, "anna.toml"), "memory = \"base.db\"\n")
	writeFile(t, filepath.Join(dir, "anna.local.toml"), "memory = \"local.db\"\n")

	// Act
	stdout, stderr, err := executeCommand("settings")

	// Assert
	if err != nil {
		t.Fatalf("settings failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "memory\tlocal.db\tfile") {
		t.Fatalf("stdout = %q, want the local config", stdout)
	}
}

func TestWhenAParentDirectoryHasAConfigTheWorkingDirectoryUsesIt(t *testing.T) {
	// Arrange
	parent := tempDir(t)
	child := filepath.Join(parent, "child")
	if err := os.Mkdir(child, 0o700); err != nil {
		t.Fatalf("mkdir child: %v", err)
	}
	writeFile(t, filepath.Join(parent, "anna.toml"), "memory = \"parent.db\"\n")
	t.Chdir(child)
	deps := testDependencies(defaultTestDependencies())
	deps.ConfigStopDir = parent

	// Act
	stdout, stderr, err := executeCommandWithDependencies(deps, "settings")

	// Assert
	if err != nil {
		t.Fatalf("settings failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "memory\tparent.db\tfile") {
		t.Fatalf("stdout = %q, want the parent config", stdout)
	}
}

func TestWhenAChildConfigExistsItOverridesTheParentConfig(t *testing.T) {
	// Arrange
	parent := tempDir(t)
	child := filepath.Join(parent, "child")
	if err := os.Mkdir(child, 0o700); err != nil {
		t.Fatalf("mkdir child: %v", err)
	}
	writeFile(t, filepath.Join(parent, "anna.toml"), "memory = \"parent.db\"\n")
	writeFile(t, filepath.Join(child, "anna.toml"), "memory = \"child.db\"\n")
	t.Chdir(child)
	deps := testDependencies(defaultTestDependencies())
	deps.ConfigStopDir = parent

	// Act
	stdout, stderr, err := executeCommandWithDependencies(deps, "settings")

	// Assert
	if err != nil {
		t.Fatalf("settings failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "memory\tchild.db\tfile") {
		t.Fatalf("stdout = %q, want the child config", stdout)
	}
}

func TestWhenAProjectConfigSetsAnAPIKeyNremRefuses(t *testing.T) {
	// Arrange
	dir := tempDir(t)
	t.Chdir(dir)
	writeFile(t, filepath.Join(dir, "anna.toml"), "[embedder]\napi-key = \"secret\"\n")

	// Act
	_, _, err := executeCommand("settings")

	// Assert
	if ExitCode(err) != 2 || !strings.Contains(err.Error(), "global-only") {
		t.Fatalf("error = %v, want a global-only usage error", err)
	}
}

func TestWhenTheConfigEnvironmentVariablePointsAtAFileNremUsesIt(t *testing.T) {
	// Arrange
	dir := tempDir(t)
	t.Chdir(dir)
	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nANNA_CONFIG should select the memory name.\n")
	configPath := filepath.Join(dir, "picked.toml")
	writeFile(t, configPath, "memory = \"picked.db\"\n")
	t.Setenv("ANNA_CONFIG", configPath)

	// Act
	stdout, stderr, err := executeCommand("nrem", source)

	// Assert
	if err != nil {
		t.Fatalf("nrem failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, filepath.Join(source, "picked.db")) {
		t.Fatalf("stdout = %q, want the memory from ANNA_CONFIG", stdout)
	}
}

func TestWhenXDGConfigHomeIsRelativeItIsIgnored(t *testing.T) {
	// Arrange
	dir := tempDir(t)
	t.Chdir(dir)
	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nA relative XDG path must not be read.\n")
	relative := filepath.Join(dir, "relative")
	if err := os.MkdirAll(filepath.Join(relative, "anna"), 0o700); err != nil {
		t.Fatalf("mkdir relative config: %v", err)
	}
	writeFile(t, filepath.Join(relative, "anna", "config.toml"), "memory = \"relative.db\"\n")
	home := filepath.Join(tempDir(t), "home")
	if err := os.MkdirAll(filepath.Join(home, ".config", "anna"), 0o700); err != nil {
		t.Fatalf("mkdir home config: %v", err)
	}
	writeFile(t, filepath.Join(home, ".config", "anna", "config.toml"), "memory = \"home.db\"\n")
	t.Setenv("XDG_CONFIG_HOME", "relative")
	t.Setenv("HOME", home)

	// Act
	stdout, stderr, err := executeCommandWithDependencies(defaultTestDependencies(), "nrem", source)

	// Assert
	if err != nil {
		t.Fatalf("nrem failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, filepath.Join(source, "home.db")) {
		t.Fatalf("stdout = %q, want the home config rather than the relative XDG path", stdout)
	}
}

func TestWhenDebugIsSetAnUnexpectedErrorWritesAnANSIFreeLog(t *testing.T) {
	// Arrange
	state := tempDir(t)
	t.Setenv("XDG_STATE_HOME", state)
	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nDebug should capture the tokenizer failure.\n")

	// Act
	_, stderr, err := executeCommandWithDependencies(testDependencies(Dependencies{
		TextSource: defaultTestDependencies().TextSource,
		IndexStore: defaultTestDependencies().IndexStore,
		NewEmbedder: func(EmbedderSettings) (core.Embedder, error) {
			return fakeEmbedder{}, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			return nil, errors.New("boom \x1b[31mred\x1b[0m")
		},
	}), "nrem", source, "--debug")

	// Assert
	if ExitCode(err) != 1 {
		t.Fatalf("exit = %d, error = %v, want failure exit 1", ExitCode(err), err)
	}
	if !strings.Contains(err.Error(), "goroutine") || !strings.Contains(err.Error(), "debug log:") {
		t.Fatalf("stderr = %q, error = %q, want a traceback and a log path", stderr, err)
	}
	matches, globErr := filepath.Glob(filepath.Join(state, "anna", "anna-*.log"))
	if globErr != nil || len(matches) != 1 {
		t.Fatalf("logs = %v, err = %v, want one debug log", matches, globErr)
	}
	body, readErr := os.ReadFile(matches[0])
	if readErr != nil {
		t.Fatalf("read log: %v", readErr)
	}
	if strings.Contains(string(body), "\x1b") {
		t.Fatalf("log = %q, want no ANSI escapes", body)
	}
}

func TestWhenASignalIsReraisedTheProcessEndsWithThatSignal(t *testing.T) {
	if os.Getenv("ANNA_RERAISE") == "1" {
		Reraise(syscall.SIGTERM)
		os.Exit(1)
	}

	// Act
	bin, err := os.Executable()
	if err != nil {
		t.Fatalf("executable: %v", err)
	}
	cmd := exec.Command(bin, "-test.run=^TestWhenASignalIsReraisedTheProcessEndsWithThatSignal$")
	cmd.Env = append(os.Environ(), "ANNA_RERAISE=1")
	err = cmd.Run()

	// Assert
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("process error = %v, want death by signal", err)
	}
	status, ok := exitErr.Sys().(syscall.WaitStatus)
	if !ok || !status.Signaled() || status.Signal() != syscall.SIGTERM {
		t.Fatalf("status = %#v, want SIGTERM", exitErr.Sys())
	}
}

func TestWhenStdoutIsClosedTheCommandSucceeds(t *testing.T) {
	t.Parallel()

	// Arrange
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	if err := reader.Close(); err != nil {
		t.Fatalf("close pipe: %v", err)
	}
	cmd := NewRootCommand("dev", testDependencies(defaultTestDependencies()))
	cmd.SetOut(writer)
	cmd.SetErr(writer)
	cmd.SetArgs([]string{"--help"})

	// Act
	err = Execute(cmd)

	// Assert
	if err != nil {
		t.Fatalf("closed stdout error = %v, want success", err)
	}
}
