package reqconv

import (
	"crypto/sha256"
	"encoding/hex"
	"maps"
	"strings"
)

const maxToolNameLen = 64

// ToolNameMap provides bidirectional mapping between original and shortened tool names.
// All methods are safe to call on a nil receiver (no-op passthrough).
type ToolNameMap struct {
	toShort    map[string]string
	toOriginal map[string]string
}

// NewToolNameMap creates a new ToolNameMap. Maps are lazily allocated on first use.
func NewToolNameMap() *ToolNameMap {
	return &ToolNameMap{}
}

// Shorten returns a Kiro-valid tool name (ASCII letters, digits, underscores and
// hyphens, at most 64 bytes). Names that need normalization or shortening are
// suffixed with a hash of the original so they cannot collide with a valid name.
// Safe to call on nil receiver.
func (m *ToolNameMap) Shorten(name string) string {
	if m == nil || (len(name) <= maxToolNameLen && isKiroToolName(name)) {
		return name
	}
	if short, ok := m.toShort[name]; ok {
		return short
	}
	if m.toShort == nil {
		m.toShort = make(map[string]string)
		m.toOriginal = make(map[string]string)
	}
	prefix := sanitizeToolName(name)
	if len(prefix) > 50 {
		prefix = prefix[:50]
	}
	if prefix == "" {
		prefix = "tool"
	}
	h := sha256.Sum256([]byte(name))
	short := prefix + "_" + hex.EncodeToString(h[:])[:13]
	m.toShort[name] = short
	m.toOriginal[short] = name
	return short
}

func isKiroToolName(name string) bool {
	if name == "" {
		return false
	}
	for _, r := range name {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-') {
			return false
		}
	}
	return true
}

func sanitizeToolName(name string) string {
	var b strings.Builder
	b.Grow(len(name))
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	return b.String()
}

// Restore returns the original name for a shortened name, or name itself if not mapped.
// Safe to call on nil receiver.
func (m *ToolNameMap) Restore(name string) string {
	if m == nil {
		return name
	}
	if orig, ok := m.toOriginal[name]; ok {
		return orig
	}
	return name
}

// ReverseMap returns a copy of the short→original map for use in the response path.
// Returns nil if no mappings exist. A copy is returned so callers can freely
// mutate the result without affecting subsequent Shorten calls.
func (m *ToolNameMap) ReverseMap() map[string]string {
	if m == nil || len(m.toOriginal) == 0 {
		return nil
	}
	out := make(map[string]string, len(m.toOriginal))
	maps.Copy(out, m.toOriginal)
	return out
}
