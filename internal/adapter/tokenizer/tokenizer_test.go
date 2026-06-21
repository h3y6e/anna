package tokenizer

import (
	"slices"
	"testing"
)

func TestKagomeTokenizesJapaneseCompoundsForSearch(t *testing.T) {
	t.Parallel()

	tokenizer, err := New()
	if err != nil {
		t.Fatalf("New error = %v", err)
	}

	documentTokens, err := tokenizer.TokenizeDocument(t.Context(), "# 投票作成UI\n選択肢編集")
	if err != nil {
		t.Fatalf("TokenizeDocument error = %v", err)
	}
	for _, want := range []string{"投票", "作成", "選択", "編集"} {
		if !slices.Contains(documentTokens, want) {
			t.Fatalf("document tokens = %#v, want %q", documentTokens, want)
		}
	}

	queryTokens, err := tokenizer.TokenizeQuery(t.Context(), "投票 作成")
	if err != nil {
		t.Fatalf("TokenizeQuery error = %v", err)
	}
	for _, want := range []string{"投票", "作成"} {
		if !slices.Contains(queryTokens, want) {
			t.Fatalf("query tokens = %#v, want %q", queryTokens, want)
		}
	}
}

func TestKagomeTokenizeDocumentPreservesTermFrequency(t *testing.T) {
	t.Parallel()

	tokenizer, err := New()
	if err != nil {
		t.Fatalf("New error = %v", err)
	}

	tokens, err := tokenizer.TokenizeDocument(t.Context(), "投票 投票")
	if err != nil {
		t.Fatalf("TokenizeDocument error = %v", err)
	}
	if got := countToken(tokens, "投票"); got != 2 {
		t.Fatalf("document token frequency = %d, want 2: %#v", got, tokens)
	}
}

func TestKagomeTokenizeQueryDeduplicatesTerms(t *testing.T) {
	t.Parallel()

	tokenizer, err := New()
	if err != nil {
		t.Fatalf("New error = %v", err)
	}

	tokens, err := tokenizer.TokenizeQuery(t.Context(), "投票 投票")
	if err != nil {
		t.Fatalf("TokenizeQuery error = %v", err)
	}
	if got := countToken(tokens, "投票"); got != 1 {
		t.Fatalf("query token frequency = %d, want 1: %#v", got, tokens)
	}
}

func countToken(tokens []string, token string) int {
	var count int
	for _, candidate := range tokens {
		if candidate == token {
			count++
		}
	}
	return count
}
