package tokenizer

import (
	"fmt"
	"strings"
	"unicode"

	"github.com/ikawaha/kagome-dict/uni"
	kagome "github.com/ikawaha/kagome/v2/tokenizer"
)

type Kagome struct {
	tokenizer *kagome.Tokenizer
}

func New() (*Kagome, error) {
	t, err := kagome.New(uni.Dict(), kagome.OmitBosEos())
	if err != nil {
		return nil, fmt.Errorf("create kagome tokenizer: %w", err)
	}
	return &Kagome{tokenizer: t}, nil
}

func (t *Kagome) Tokenize(text string) []string {
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

func hasLetterOrDigit(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}
