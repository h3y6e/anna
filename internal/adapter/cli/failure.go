package cli

import (
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

const docsConfig = "https://github.com/h3y6e/anna#configuration"

const docsSchema = "https://github.com/h3y6e/anna/blob/main/schema/config.schema.json"

type failure struct {
	Exit  int
	Code  string
	Cause string
	Fix   string
	Docs  string
	Log   string
	Trace string
	err   error
}

func (f *failure) Error() string {
	msg := f.Cause
	if f.Fix != "" && !strings.Contains(msg, f.Fix) {
		msg += "\n" + f.Fix
	}
	if f.Docs != "" && !strings.Contains(msg, f.Docs) {
		msg += "\n" + f.Docs
	}
	if f.Log != "" {
		msg += "\ndebug log: " + f.Log
	}
	if f.Trace != "" {
		msg += "\n" + f.Trace
	}
	return msg
}

func (f *failure) Unwrap() error { return f.err }

func usageFailure(cmd *cobra.Command, cause string, err error) *failure {
	return &failure{Exit: 2, Code: "usage", Cause: cause, Fix: helpFix(cmd), err: err}
}

func helpFix(cmd *cobra.Command) string {
	if cmd == nil || cmd.CommandPath() == "" {
		return "anna --help"
	}
	return cmd.CommandPath() + " --help"
}

func ExitCode(err error) int {
	if err == nil {
		return 0
	}
	if f, ok := errors.AsType[*failure](err); ok && f.Exit != 0 {
		return f.Exit
	}
	return 1
}

// Reraise sends sig to this process after clearing the CLI's signal handlers.
func Reraise(sig os.Signal) {
	signal.Reset(os.Interrupt, syscall.SIGTERM)
	s, ok := sig.(syscall.Signal)
	if !ok || s == 0 {
		return
	}
	_ = syscall.Kill(syscall.Getpid(), s)
	time.Sleep(time.Second)
}

func classify(cmd *cobra.Command, cfg *viper.Viper, err error) error {
	if err == nil {
		return nil
	}
	if _, ok := asFailure(err); ok {
		return err
	}
	if isUsageText(err.Error()) {
		return usageFailure(cmd, err.Error(), err)
	}
	f := &failure{Exit: 1, Code: "internal", Cause: err.Error(), err: err}
	if cfg != nil && cfg.GetBool("debug") {
		f.Trace = stripANSI(string(debug.Stack()))
		if path, logErr := writeDebugLog(f.Cause, f.Trace); logErr == nil {
			f.Log = path
		}
	} else {
		f.Fix = "re-run with --debug"
	}
	return f
}

func isUsageText(msg string) bool {
	for _, needle := range []string{
		"unknown command",
		"unknown flag",
		"unknown shorthand",
		"flag needs",
		"invalid argument",
		"arg(s)",
		"required flag",
	} {
		if strings.Contains(msg, needle) {
			return true
		}
	}
	return false
}

func writeDebugLog(cause string, trace string) (string, error) {
	dir, err := stateDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	path := filepath.Join(dir, "anna-"+time.Now().UTC().Format("20060102T150405.000Z")+".log")
	body := stripANSI(cause + "\n\n" + trace)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		return "", err
	}
	return path, nil
}

func stateDir() (string, error) {
	if xdg := os.Getenv("XDG_STATE_HOME"); filepath.IsAbs(xdg) {
		return filepath.Join(xdg, "anna"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home directory: %w", err)
	}
	return filepath.Join(home, ".local", "state", "anna"), nil
}

func asFailure(err error) (*failure, bool) {
	return errors.AsType[*failure](err)
}

var ansiPattern = regexp.MustCompile(`\x1b\[[0-9;]*[A-Za-z]`)

func stripANSI(s string) string {
	return ansiPattern.ReplaceAllString(s, "")
}
