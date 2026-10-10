package cli

import (
	"encoding/json"
	"io"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

const outputSchema = 1

type resultRecord struct {
	Record string `json:"record"`
	Schema int    `json:"schema"`
	OK     bool   `json:"ok"`
	Code   string `json:"code,omitempty"`
	Cause  string `json:"cause,omitempty"`
	Fix    string `json:"fix,omitempty"`
	Docs   string `json:"docs,omitempty"`
	Log    string `json:"log,omitempty"`
}

func writeSuccess(w io.Writer) error {
	return json.NewEncoder(w).Encode(resultRecord{Record: "result", Schema: outputSchema, OK: true})
}

func writeFailure(w io.Writer, err error) error {
	f, ok := asFailure(err)
	if !ok {
		f = &failure{Code: "internal", Cause: err.Error()}
	}
	return json.NewEncoder(w).Encode(resultRecord{
		Record: "result",
		Schema: outputSchema,
		OK:     false,
		Code:   f.Code,
		Cause:  f.Cause,
		Fix:    f.Fix,
		Docs:   f.Docs,
		Log:    f.Log,
	})
}

func jsonEnabled(cfg *viper.Viper, cmd *cobra.Command) bool {
	if cfg != nil && cfg.GetBool("json") {
		return true
	}
	if cmd == nil {
		return false
	}
	flag := cmd.Flags().Lookup("json")
	return flag != nil && flag.Changed && flag.Value.String() == "true"
}
