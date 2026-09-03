FROM golang:1.27-alpine AS build

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o /out/kiro-proxy ./cmd/kiro-proxy

FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=build /out/kiro-proxy /usr/local/bin/kiro-proxy

EXPOSE 3456

ENTRYPOINT ["/usr/local/bin/kiro-proxy"]
