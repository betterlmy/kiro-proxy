
BIN := dist/kiro-proxy

.PHONY: build install run debug test test-e2e lint vet fmt fix clean

build:
	go build -o $(BIN) ./cmd/kiro-proxy

install:
	go install ./cmd/kiro-proxy

run:
	go run ./cmd/kiro-proxy $(ARGS)

debug:
	go run ./cmd/kiro-proxy -debug $(ARGS)

test:
	go test -race ./...

test-e2e:
	go test -tags e2e -race -timeout 120s ./internal/e2e/

lint:
	golangci-lint run

vet:
	go vet ./...

fmt:
	golangci-lint fmt

fix:
	go fix ./...

clean:
	rm -f $(BIN)
