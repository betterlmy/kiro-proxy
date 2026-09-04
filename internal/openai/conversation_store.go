package openai

import (
	"sync"
	"time"
)

const (
	defaultConversationTTL      = 30 * time.Minute
	defaultConversationCapacity = 1024
)

type conversationEntry struct {
	conversationID string
	expiresAt      time.Time
	lastAccess     time.Time
}

// conversationStore retains only the opaque response-ID to Kiro-conversation
// mapping needed by Responses continuations. It deliberately has no disk
// persistence: a proxy restart still requires the client to resend history.
type conversationStore struct {
	mu       sync.Mutex
	entries  map[string]conversationEntry
	ttl      time.Duration
	capacity int
	now      func() time.Time
}

func newConversationStore(ttl time.Duration, capacity int, now func() time.Time) *conversationStore {
	if capacity < 1 {
		capacity = 1
	}
	if ttl <= 0 {
		ttl = defaultConversationTTL
	}
	if now == nil {
		now = time.Now
	}
	return &conversationStore{entries: make(map[string]conversationEntry), ttl: ttl, capacity: capacity, now: now}
}

func defaultConversationStore() *conversationStore {
	return newConversationStore(defaultConversationTTL, defaultConversationCapacity, time.Now)
}

func (s *conversationStore) Get(responseID string) (string, bool) {
	if responseID == "" {
		return "", true
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	s.removeExpired(now)
	entry, ok := s.entries[responseID]
	if !ok {
		return "", false
	}
	entry.lastAccess = now
	s.entries[responseID] = entry
	return entry.conversationID, true
}

func (s *conversationStore) Put(responseID, conversationID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	s.removeExpired(now)
	if _, exists := s.entries[responseID]; !exists && len(s.entries) >= s.capacity {
		s.evictLeastRecentlyUsed()
	}
	s.entries[responseID] = conversationEntry{conversationID: conversationID, expiresAt: now.Add(s.ttl), lastAccess: now}
}

func (s *conversationStore) removeExpired(now time.Time) {
	for id, entry := range s.entries {
		if !entry.expiresAt.After(now) {
			delete(s.entries, id)
		}
	}
}

func (s *conversationStore) evictLeastRecentlyUsed() {
	var oldestID string
	var oldest time.Time
	for id, entry := range s.entries {
		if oldestID == "" || entry.lastAccess.Before(oldest) || (entry.lastAccess.Equal(oldest) && id < oldestID) {
			oldestID = id
			oldest = entry.lastAccess
		}
	}
	if oldestID != "" {
		delete(s.entries, oldestID)
	}
}
