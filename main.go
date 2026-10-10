package main

import (
	"fmt"
	"os"

	"github.com/h3y6e/anna/cmd"
	"github.com/h3y6e/anna/internal/adapter/cli"
)

var version = "dev"

func main() {
	if err := cmd.Execute(version); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(cli.ExitCode(err))
	}
}
