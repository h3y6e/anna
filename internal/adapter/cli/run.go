package cli

import (
	"errors"
	"sync"
	"syscall"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

type runtimeConfig struct {
	cfg   *viper.Viper
	files []string
}

var commandRuntime sync.Map

func Execute(root *cobra.Command) error {
	defer commandRuntime.Delete(root)
	found, err := root.ExecuteC()
	if errors.Is(err, syscall.EPIPE) {
		return nil
	}
	if err == nil {
		return nil
	}
	rt, _ := commandRuntime.Load(root)
	var cfg *viper.Viper
	if rt, ok := rt.(*runtimeConfig); ok {
		cfg = rt.cfg
	}
	err = classify(found, cfg, err)
	if jsonEnabled(cfg, found) {
		_ = writeFailure(found.OutOrStdout(), err)
	}
	return err
}
