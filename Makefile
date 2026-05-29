VERSION ?= dev
COMMIT  := $(shell git rev-parse --short HEAD 2>/dev/null || echo "none")
DATE    := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
LDFLAGS := -ldflags "-X github.com/sovereignty-labs/nollama/internal/version.Version=$(VERSION) \
                      -X github.com/sovereignty-labs/nollama/internal/version.Commit=$(COMMIT) \
                      -X github.com/sovereignty-labs/nollama/internal/version.Date=$(DATE)"

.PHONY: build clean test package-runtime

build:
	go build $(LDFLAGS) -o nollama ./cmd/nollama

clean:
	rm -f nollama

test:
	go test ./...

install: build
	install -m 755 nollama /usr/local/bin/nollama

package-runtime:
	sh scripts/package-runtime.sh "$(SOURCE_RUNTIME_DIR)" "$(LLAMACPP_VERSION)" "$(BACKEND)"

.DEFAULT_GOAL := build
