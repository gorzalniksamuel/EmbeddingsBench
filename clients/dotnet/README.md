# EmbeddingsBench .NET client

A small .NET 8 library for the EmbeddingsBench inference API (`web/inference.py`): classify texts or
precomputed embeddings with any registry model and get the label, confidence, top-k and the full class
distribution; or fetch embeddings to store and classify later.

## Use it in an ASP.NET Core / generic-host application

Reference the project (or the package built with `dotnet pack`):

```bash
dotnet add reference path/to/clients/dotnet/EmbeddingsBench.Client/EmbeddingsBench.Client.csproj
```

```csharp
using EmbeddingsBench.Client;

builder.Services.AddEmbeddingsBench(o =>
{
    o.BaseAddress = new Uri("http://gpu-host:8096");          // where web/inference.py runs
    o.ApiKey = builder.Configuration["EmbeddingsBench:ApiKey"]; // if the service runs with EB_API_KEY
});

// anywhere via constructor injection
public sealed class TicketRouter(IEmbeddingsBenchClient bench)
{
    public async Task<string> RouteAsync(string text, CancellationToken ct)
    {
        var p = await bench.PredictOneAsync("support-tickets-logreg", text, topK: 3, cancellationToken: ct);
        return p.Confidence >= 0.6 ? p.LabelId : "manual-review";
    }
}
```

Without dependency injection: `var client = new EmbeddingsBenchClient(new EmbeddingsBenchOptions { BaseAddress = new("http://gpu-host:8096") });`

## API

| Method | Purpose |
|---|---|
| `ListModelsAsync()` | ready models: `Model` (`name:vN`), `Method`, `Dataset`, `Input`, `TestAccuracy`, `NeedsTexts` |
| `PredictAsync(model, texts, topK, distribution)` | classify texts; `model` is `name` (latest ready version) or `name:version` |
| `PredictAsync(model, embeddings, topK, distribution)` | classify 1,024-d embeddings (models on frozen embeddings only) |
| `PredictOneAsync(model, text)` | convenience for a single text |
| `EmbedAsync(texts, maxTokens)` | L2-normalised embeddings from the benchmark encoder |
| `GetHealthAsync()` | device and configuration of the service |

Each `Prediction` has `Label` (readable), `LabelId` (dataset id), `Index`, `Confidence`, `TopK` and, with
`distribution: true`, `Distribution` (probability per class). `PredictResponse.Scores` says whether the numbers are
calibrated probabilities or softmaxed scores (SVM, zero-shot). Errors from the service raise
`EmbeddingsBenchException` with the HTTP `StatusCode` (400 bad input, 401 token, 404 unknown model).

## Generating your own client

The service publishes its OpenAPI 3 description at `/openapi.yaml` (also [`clients/openapi.yaml`](../openapi.yaml)),
so NSwag, Kiota or Visual Studio's "Connected Services" can generate a client in any style you prefer.

## Example

```bash
EB_INFER_URL=http://gpu-host:8096 EB_API_KEY=secret dotnet run --project Example -- banking-logreg "My card has not arrived yet"
```
