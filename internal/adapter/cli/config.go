package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/spf13/viper"
)

type configSearch struct {
	file   string
	local  []string
	global []string
	stop   string
}

type readConfigFile struct {
	path         string
	definesNotes bool
}

func configSearchFrom(deps Dependencies, file string) configSearch {
	search := configSearch{file: file, stop: deps.ConfigStopDir}
	if cwd, err := os.Getwd(); err == nil && cwd != "" {
		search.local = append(search.local, cwd)
	}
	if deps.ConfigSearchPaths != nil {
		search.global = append(search.global, deps.ConfigSearchPaths...)
	} else {
		search.global = append(search.global, defaultConfigSearchPaths()...)
	}
	return search
}

func readConfig(cfg *viper.Viper, search configSearch) ([]string, error) {
	read, err := readConfigFiles(cfg, search)
	if err != nil {
		return nil, err
	}
	if err := resolveRelativeNotes(cfg, read); err != nil {
		return nil, err
	}
	files := make([]string, 0, len(read))
	for _, file := range read {
		files = append(files, file.path)
	}
	return files, nil
}

func resolveRelativeNotes(cfg *viper.Viper, files []readConfigFile) error {
	if _, fromEnv := os.LookupEnv("ANNA_NOTES"); fromEnv {
		return nil
	}
	base := ""
	for _, file := range files {
		if file.definesNotes {
			base = filepath.Dir(file.path)
		}
	}
	if base == "" {
		return nil
	}
	dirs, err := configuredNotes(cfg)
	if err != nil {
		return err
	}
	resolved := make([]string, 0, len(dirs))
	for _, dir := range dirs {
		expanded, err := expandPath(dir)
		if err != nil {
			return err
		}
		if !filepath.IsAbs(expanded) {
			expanded = filepath.Join(base, expanded)
		}
		resolved = append(resolved, expanded)
	}
	cfg.Set("notes", resolved)
	return nil
}

func readConfigFiles(cfg *viper.Viper, search configSearch) ([]readConfigFile, error) {
	if search.file != "" {
		path, err := expandPath(search.file)
		if err != nil {
			return nil, err
		}
		read, err := applyConfigFile(cfg, path, isGlobalConfig(path, search.global))
		if err != nil {
			return nil, err
		}
		return []readConfigFile{read}, nil
	}

	var files []readConfigFile
	for _, name := range []string{"config.toml", "anna.toml"} {
		path := firstExistingFile(search.global, name)
		if path == "" {
			continue
		}
		read, err := applyConfigFile(cfg, path, true)
		if err != nil {
			return nil, err
		}
		files = append(files, read)
	}

	for _, dir := range projectDirs(search.local, search.stop) {
		for _, path := range projectFiles(dir) {
			read, err := applyConfigFile(cfg, path, false)
			if err != nil {
				return nil, err
			}
			files = append(files, read)
		}
	}
	return files, nil
}

func applyConfigFile(cfg *viper.Viper, path string, allowSecrets bool) (readConfigFile, error) {
	file, err := openConfig(path)
	if err != nil {
		return readConfigFile{}, configFailure(fmt.Sprintf("read config %s: %s", path, err.Error()), err)
	}
	if !allowSecrets && file.IsSet("embedder.api-key") {
		return readConfigFile{}, &failure{
			Exit:  2,
			Code:  "config",
			Cause: fmt.Sprintf("embedder.api-key in %s is global-only", path),
			Fix:   "set ANNA_EMBEDDER_API_KEY or embedder.api-key in the global config",
			Docs:  docsConfig,
		}
	}
	if err := cfg.MergeConfigMap(file.AllSettings()); err != nil {
		return readConfigFile{}, configFailure(fmt.Sprintf("read config %s: %s", path, err.Error()), err)
	}
	return readConfigFile{path: path, definesNotes: file.IsSet("notes")}, nil
}

func openConfig(path string) (*viper.Viper, error) {
	file := viper.New()
	file.SetConfigFile(path)
	file.SetConfigType("toml")
	if err := file.ReadInConfig(); err != nil {
		return nil, err
	}
	return file, nil
}

func configFailure(cause string, err error) *failure {
	return &failure{Exit: 2, Code: "config", Cause: cause, Fix: "anna --help", Docs: docsConfig, err: err}
}

func projectDirs(starts []string, stop string) []string {
	dirs := []string{}
	for _, start := range starts {
		up := []string{}
		dir := start
		for {
			up = append(up, dir)
			if stop != "" && dir == stop {
				break
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			if stop != "" && !contains(stop, parent) {
				break
			}
			dir = parent
		}
		for _, dir := range slices.Backward(up) {
			dirs = append(dirs, dir)
		}
	}
	return dirs
}

func projectFiles(dir string) []string {
	files := []string{}
	for _, name := range []string{"anna.toml", "anna.local.toml"} {
		path := filepath.Join(dir, name)
		if regularFile(path) {
			files = append(files, path)
		}
	}
	return files
}

func firstExistingFile(dirs []string, name string) string {
	for _, dir := range dirs {
		path := filepath.Join(dir, name)
		if regularFile(path) {
			return path
		}
	}
	return ""
}

func regularFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func isGlobalConfig(path string, globalPaths []string) bool {
	abs, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	for _, dir := range globalPaths {
		root, err := filepath.Abs(dir)
		if err != nil {
			continue
		}
		if contains(root, abs) {
			return true
		}
	}
	return false
}

func defaultConfigSearchPaths() []string {
	paths := make([]string, 0, 2)
	if xdgConfigHome := os.Getenv("XDG_CONFIG_HOME"); filepath.IsAbs(xdgConfigHome) {
		paths = append(paths, filepath.Join(xdgConfigHome, "anna"))
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		path := filepath.Join(home, ".config", "anna")
		if len(paths) == 0 || paths[len(paths)-1] != path {
			paths = append(paths, path)
		}
	}
	return paths
}

func expandPath(path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	if path == "~" || strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve home directory: %w", err)
		}
		if path == "~" {
			return home, nil
		}
		return filepath.Join(home, strings.TrimPrefix(path, "~/")), nil
	}
	return path, nil
}
