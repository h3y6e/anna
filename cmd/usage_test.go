package cmd

import (
	"strings"
	"testing"

	"github.com/h3y6e/anna/internal/adapter/cli"
	cobrausage "github.com/jdx/usage/integrations/cobra"
)

func TestWhenTheUsageSpecIsGeneratedItDeclaresNremAndRecall(t *testing.T) {
	t.Parallel()

	// Act
	spec := cobrausage.Generate(cli.NewRootCommand("dev", cli.Dependencies{}))

	// Assert
	if !strings.Contains(spec, "nrem") || !strings.Contains(spec, "recall") || !strings.Contains(spec, "settings") {
		t.Fatalf("spec = %q, want nrem, recall, and settings", spec)
	}
}
