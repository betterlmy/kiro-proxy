package openai

import (
	"testing"
	"time"
)

func TestConversationStoreExpiresEntries(t *testing.T) {
	now := time.Date(2026, time.September, 4, 12, 0, 0, 0, time.UTC)
	store := newConversationStore(time.Minute, 2, func() time.Time { return now })
	store.Put("resp_1", "conv_1")
	if got, ok := store.Get("resp_1"); !ok || got != "conv_1" {
		t.Fatalf("Get() = %q, %t", got, ok)
	}
	now = now.Add(time.Minute)
	if _, ok := store.Get("resp_1"); ok {
		t.Fatal("expired response remained available")
	}
}

func TestConversationStoreEvictsLeastRecentlyUsed(t *testing.T) {
	now := time.Date(2026, time.September, 4, 12, 0, 0, 0, time.UTC)
	store := newConversationStore(time.Hour, 2, func() time.Time { return now })
	store.Put("resp_1", "conv_1")
	now = now.Add(time.Second)
	store.Put("resp_2", "conv_2")
	now = now.Add(time.Second)
	if _, ok := store.Get("resp_1"); !ok {
		t.Fatal("resp_1 missing before eviction")
	}
	now = now.Add(time.Second)
	store.Put("resp_3", "conv_3")
	if _, ok := store.Get("resp_1"); !ok {
		t.Fatal("most recently used entry was evicted")
	}
	if _, ok := store.Get("resp_2"); ok {
		t.Fatal("least recently used entry was not evicted")
	}
}
