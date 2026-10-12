using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Extensions.Options;

namespace EmbeddingsBench.Client;

/// <summary>Settings for <see cref="EmbeddingsBenchClient"/>.</summary>
public sealed class EmbeddingsBenchOptions
{
    /// <summary>Base address of the inference service (web/inference.py), e.g. http://gpu-host:8096.</summary>
    public Uri BaseAddress { get; set; } = new("http://localhost:8096");

    /// <summary>Bearer token, if the service runs with EB_API_KEY.</summary>
    public string? ApiKey { get; set; }

    /// <summary>Request timeout; embedding long texts on a busy GPU can take a while.</summary>
    public TimeSpan Timeout { get; set; } = TimeSpan.FromMinutes(5);
}

/// <summary>Text classification and embeddings via the EmbeddingsBench inference API.</summary>
public interface IEmbeddingsBenchClient
{
    /// <summary>Device and configuration of the service.</summary>
    Task<HealthResponse> GetHealthAsync(CancellationToken cancellationToken = default);

    /// <summary>All ready models (name:version, method, dataset, required input, test accuracy).</summary>
    Task<IReadOnlyList<ModelInfo>> ListModelsAsync(CancellationToken cancellationToken = default);

    /// <summary>Classify texts. <paramref name="model"/> is "name" (latest ready version) or "name:version".</summary>
    Task<PredictResponse> PredictAsync(string model, IEnumerable<string> texts, int topK = 3, bool distribution = false,
        CancellationToken cancellationToken = default);

    /// <summary>Classify precomputed embeddings (1,024-d, from <see cref="EmbedAsync"/> or the same model elsewhere).</summary>
    Task<PredictResponse> PredictAsync(string model, IEnumerable<float[]> embeddings, int topK = 3, bool distribution = false,
        CancellationToken cancellationToken = default);

    /// <summary>Classify a single text and return its prediction.</summary>
    Task<Prediction> PredictOneAsync(string model, string text, int topK = 3, bool distribution = false,
        CancellationToken cancellationToken = default);

    /// <summary>Embeddings of the texts (L2-normalised, 1,024 dimensions), e.g. to store and classify later.</summary>
    Task<IReadOnlyList<float[]>> EmbedAsync(IEnumerable<string> texts, int maxTokens = 2048, CancellationToken cancellationToken = default);
}

/// <summary>Raised when the service answers with an error status (400 bad input, 401 token, 404 unknown model, 5xx).</summary>
public sealed class EmbeddingsBenchException(int statusCode, string message) : Exception(message)
{
    /// <summary>HTTP status code returned by the service.</summary>
    public int StatusCode { get; } = statusCode;
}

/// <inheritdoc />
public sealed class EmbeddingsBenchClient : IEmbeddingsBenchClient
{
    private const int EmbeddingDimensions = 1024;
    private readonly HttpClient _http;

    /// <summary>Use with dependency injection (see <c>AddEmbeddingsBench</c>); the HttpClient is configured there.</summary>
    public EmbeddingsBenchClient(HttpClient http) => _http = http;

    /// <summary>Use without dependency injection.</summary>
    public EmbeddingsBenchClient(EmbeddingsBenchOptions options) : this(Configure(new HttpClient(), options)) { }

    internal EmbeddingsBenchClient(HttpClient http, IOptions<EmbeddingsBenchOptions> options) : this(Configure(http, options.Value)) { }

    internal static HttpClient Configure(HttpClient http, EmbeddingsBenchOptions options)
    {
        http.BaseAddress = options.BaseAddress;
        http.Timeout = options.Timeout;
        if (!string.IsNullOrEmpty(options.ApiKey))
            http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", options.ApiKey);
        return http;
    }

    /// <inheritdoc />
    public Task<HealthResponse> GetHealthAsync(CancellationToken cancellationToken = default) =>
        SendAsync<HealthResponse>(HttpMethod.Get, "health", null, cancellationToken);

    /// <inheritdoc />
    public async Task<IReadOnlyList<ModelInfo>> ListModelsAsync(CancellationToken cancellationToken = default) =>
        (await SendAsync<ModelsResponse>(HttpMethod.Get, "v1/models", null, cancellationToken)).Models;

    /// <inheritdoc />
    public Task<PredictResponse> PredictAsync(string model, IEnumerable<string> texts, int topK = 3, bool distribution = false,
        CancellationToken cancellationToken = default)
    {
        var list = texts.ToList();
        if (list.Count == 0) throw new ArgumentException("at least one text is required", nameof(texts));
        return SendAsync<PredictResponse>(HttpMethod.Post, "v1/predict", new PredictRequest(model, list, null, topK, distribution), cancellationToken);
    }

    /// <inheritdoc />
    public Task<PredictResponse> PredictAsync(string model, IEnumerable<float[]> embeddings, int topK = 3, bool distribution = false,
        CancellationToken cancellationToken = default)
    {
        var list = embeddings.ToList();
        if (list.Count == 0) throw new ArgumentException("at least one embedding is required", nameof(embeddings));
        if (list.Any(e => e.Length != EmbeddingDimensions))
            throw new ArgumentException($"embeddings must have {EmbeddingDimensions} dimensions", nameof(embeddings));
        return SendAsync<PredictResponse>(HttpMethod.Post, "v1/predict", new PredictRequest(model, null, list, topK, distribution), cancellationToken);
    }

    /// <inheritdoc />
    public async Task<Prediction> PredictOneAsync(string model, string text, int topK = 3, bool distribution = false,
        CancellationToken cancellationToken = default) =>
        (await PredictAsync(model, new[] { text }, topK, distribution, cancellationToken)).Predictions[0];

    /// <inheritdoc />
    public async Task<IReadOnlyList<float[]>> EmbedAsync(IEnumerable<string> texts, int maxTokens = 2048, CancellationToken cancellationToken = default)
    {
        var response = await SendAsync<EmbeddingsResponse>(HttpMethod.Post, "v1/embeddings", new EmbeddingsRequest(texts.ToList(), maxTokens), cancellationToken);
        return response.Data.OrderBy(d => d.Index).Select(d => d.Embedding).ToList();
    }

    private async Task<T> SendAsync<T>(HttpMethod method, string path, object? body, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(method, path);
        if (body is not null)  // buffered (Content-Length set): some servers do not accept chunked request bodies
            request.Content = new ByteArrayContent(JsonSerializer.SerializeToUtf8Bytes(body, body.GetType()))
            {
                Headers = { ContentType = new MediaTypeHeaderValue("application/json") },
            };
        using var response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            string message;
            try { message = (await response.Content.ReadFromJsonAsync<ErrorResponse>(cancellationToken).ConfigureAwait(false))?.Error ?? response.ReasonPhrase ?? "error"; }
            catch (JsonException) { message = response.ReasonPhrase ?? "error"; }
            throw new EmbeddingsBenchException((int)response.StatusCode, message);
        }
        return (await response.Content.ReadFromJsonAsync<T>(cancellationToken).ConfigureAwait(false))
               ?? throw new EmbeddingsBenchException((int)response.StatusCode, "empty response");
    }
}
