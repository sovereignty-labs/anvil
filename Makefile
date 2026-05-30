VERSION ?= dev
COMMIT  := $(shell git rev-parse --short HEAD 2>/dev/null || echo "none")
DATE    := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
LDFLAGS := -ldflags "-X github.com/sovereignty-labs/anvil/internal/version.Version=$(VERSION) \
                      -X github.com/sovereignty-labs/anvil/internal/version.Commit=$(COMMIT) \
                      -X github.com/sovereignty-labs/anvil/internal/version.Date=$(DATE)"

.PHONY: build clean test package-runtime

build:
	go build $(LDFLAGS) -o anvil ./cmd/anvil

clean:
	rm -f anvil

test:
	go test ./...

install: build
	install -m 755 anvil /usr/local/bin/anvil

package-runtime:
	sh scripts/package-runtime.sh "$(SOURCE_RUNTIME_DIR)" "$(LLAMACPP_VERSION)" "$(BACKEND)"

.DEFAULT_GOAL := build
