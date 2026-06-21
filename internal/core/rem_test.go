package core

import "testing"

func TestREMFindsEchoCandidates(t *testing.T) {
	t.Parallel()

	candidates, err := REM(&Index{Documents: []Document{
		{
			Path: "alpha.md",

			ContentHash: "same",
			Embedding:   []float64{1, 0},
		},
		{
			Path: "alpha-copy.md",

			ContentHash: "same",
			Embedding:   []float64{1, 0},
		},
		{
			Path: "beta.md",

			Embedding: []float64{0, 1},
		},
	}}, REMOptions{Focus: REMFocusEcho, Limit: 10, Threshold: 0.75})
	if err != nil {
		t.Fatalf("REM error = %v", err)
	}
	if len(candidates) != 1 {
		t.Fatalf("candidate count = %d, want 1: %#v", len(candidates), candidates)
	}
	if candidates[0].Focus != "echo" || candidates[0].LeftPath != "alpha-copy.md" || candidates[0].RightPath != "alpha.md" {
		t.Fatalf("candidate = %#v, want stable echo pair", candidates[0])
	}
}

func TestREMFindsSynapseCandidatesWithoutExistingLinks(t *testing.T) {
	t.Parallel()

	candidates, err := REM(&Index{Documents: []Document{
		{
			Path: "alpha.md",

			Content:   "Retrieval augmented generation notes.",
			Embedding: []float64{1, 0},
		},
		{
			Path: "bridge.md",

			Content:   "Local semantic search notes.",
			Embedding: []float64{0.8, 0.6},
		},
		{
			Path: "linked.md",

			Content:   "Already points to [[Alpha]].",
			Embedding: []float64{0.8, 0.6},
		},
	}}, REMOptions{Focus: REMFocusSynapse, Limit: 10, Threshold: 0.75})
	if err != nil {
		t.Fatalf("REM error = %v", err)
	}
	if len(candidates) != 1 {
		t.Fatalf("candidate count = %d, want 1: %#v", len(candidates), candidates)
	}
	if candidates[0].Focus != "synapse" || candidates[0].LeftPath != "alpha.md" || candidates[0].RightPath != "bridge.md" {
		t.Fatalf("candidate = %#v, want unlinked synapse pair", candidates[0])
	}
}

func TestREMRejectsUnsupportedFocus(t *testing.T) {
	t.Parallel()

	_, err := REM(&Index{}, REMOptions{Focus: REMFocus("cluster")})
	if err == nil {
		t.Fatal("REM error = nil, want unsupported focus")
	}
}

func TestREMAcceptsZeroThreshold(t *testing.T) {
	t.Parallel()

	candidates, err := REM(&Index{Documents: []Document{
		{
			Path: "alpha.md",

			Embedding: []float64{1, 0},
		},
		{
			Path: "beta.md",

			Embedding: []float64{0, 1},
		},
	}}, REMOptions{Focus: REMFocusSynapse, Limit: 10, Threshold: 0})
	if err != nil {
		t.Fatalf("REM error = %v", err)
	}
	if len(candidates) != 1 {
		t.Fatalf("candidate count = %d, want 1: %#v", len(candidates), candidates)
	}
}

func TestREMRejectsInvalidThreshold(t *testing.T) {
	t.Parallel()

	_, err := REM(&Index{}, REMOptions{Threshold: 1.1})
	if err == nil {
		t.Fatal("REM error = nil, want invalid threshold")
	}
}

func TestREMLimitTruncatesCandidates(t *testing.T) {
	t.Parallel()

	candidates, err := REM(&Index{Documents: []Document{
		{Path: "a.md", ContentHash: "same", Embedding: []float64{1, 0}},
		{Path: "b.md", ContentHash: "same", Embedding: []float64{1, 0}},
		{Path: "c.md", ContentHash: "same", Embedding: []float64{1, 0}},
	}}, REMOptions{Focus: REMFocusEcho, Limit: 1, Threshold: 0.75})
	if err != nil {
		t.Fatalf("REM error = %v", err)
	}
	if len(candidates) != 1 {
		t.Fatalf("candidate count = %d, want 1: %#v", len(candidates), candidates)
	}
}
