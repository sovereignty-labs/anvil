FROM golang:1.25 AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go test ./...
ARG VERSION=dev
ARG COMMIT=none
ARG DATE=unknown
RUN CGO_ENABLED=0 GOOS=linux go build \
    -trimpath \
    -ldflags "-s -w \
      -X github.com/sovereignty-labs/anvil/internal/version.Version=${VERSION} \
      -X github.com/sovereignty-labs/anvil/internal/version.Commit=${COMMIT} \
      -X github.com/sovereignty-labs/anvil/internal/version.Date=${DATE}" \
    -o /out/anvil ./cmd/anvil

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /out/anvil /usr/local/bin/anvil
ENTRYPOINT ["anvil"]
