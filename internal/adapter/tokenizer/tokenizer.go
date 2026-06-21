package tokenizer

import (
	"context"
	"fmt"
	"strings"
	"unicode"

	"github.com/h3y6e/anna/internal/core"
	"github.com/ikawaha/kagome-dict/uni"
	kagome "github.com/ikawaha/kagome/v2/tokenizer"
)

func New() (core.Tokenizer, error) {
	return NewKagome()
}

type Kagome struct {
	tokenizer *kagome.Tokenizer
}

func NewKagome() (*Kagome, error) {
	t, err := kagome.New(uni.Dict(), kagome.OmitBosEos())
	if err != nil {
		return nil, fmt.Errorf("create kagome tokenizer: %w", err)
	}
	return &Kagome{tokenizer: t}, nil
}

func (t *Kagome) TokenizeDocument(_ context.Context, text string) ([]string, error) {
	return t.tokenize(text), nil
}

func (t *Kagome) TokenizeQuery(_ context.Context, text string) ([]string, error) {
	return unique(t.tokenize(text)), nil
}

func (t *Kagome) tokenize(text string) []string {
	tokens := make([]string, 0)
	for _, token := range t.tokenizer.Analyze(text, kagome.Search) {
		if !keepKagomeToken(token) {
			continue
		}
		tokens = appendNormalized(tokens, token.Surface)
		if base, ok := token.BaseForm(); ok && base != token.Surface {
			tokens = appendNormalized(tokens, base)
		}
	}
	return tokens
}

func keepKagomeToken(token kagome.Token) bool {
	pos := token.POS()
	if len(pos) > 0 {
		switch pos[0] {
		case "助詞", "助動詞", "補助記号", "記号", "空白":
			return false
		}
	}
	return hasLetterOrDigit(token.Surface)
}

func appendNormalized(tokens []string, token string) []string {
	token = strings.TrimSpace(strings.ToLower(token))
	if token == "" || !hasLetterOrDigit(token) {
		return tokens
	}
	return append(tokens, token)
}

func unique(tokens []string) []string {
	seen := make(map[string]bool, len(tokens))
	out := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if seen[token] {
			continue
		}
		seen[token] = true
		out = append(out, token)
	}
	return out
}

func hasLetterOrDigit(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}
