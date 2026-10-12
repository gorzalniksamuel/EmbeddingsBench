using System.Text.Json.Serialization;

namespace EmbeddingsBench.Client;

/// <summary>A ready model in the registry, as listed by <c>GET /v1/models</c>.</summary>
public sealed record ModelInfo(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("version")] int Version,
    [property: JsonPropertyName("method")] string Method,
    [property: JsonPropertyName("dataset")] string Dataset,
    [property: JsonPropertyName("input")] string Input,
    [property: JsonPropertyName("stage")] string? Stage,
    [property: JsonPropertyName("test_accuracy")] double? TestAccuracy,
    [property: JsonPropertyName("description")] string? Description)
{
    /// <summary>True when the model needs texts (TF-IDF, ensembles, fine-tuned models); embeddings are not enough.</summary>
    public bool NeedsTexts => Input.Contains("texts", StringComparison.Ordinal);
}

/// <summary>One class and its probability.</summary>
public sealed record LabelScore(
    [property: JsonPropertyName("label")] string Label,
    [property: JsonPropertyName("label_id")] string LabelId,
    [property: JsonPropertyName("probability")] double Probability);

/// <summary>The prediction for one input text or embedding.</summary>
public sealed record Prediction(
    [property: JsonPropertyName("label")] string Label,
    [property: JsonPropertyName("label_id")] string LabelId,
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("confidence")] double Confidence,
    [property: JsonPropertyName("top_k")] IReadOnlyList<LabelScore> TopK,
    [property: JsonPropertyName("distribution")] IReadOnlyDictionary<string, double>? Distribution);

/// <summary>Response of <c>POST /v1/predict</c>.</summary>
public sealed record PredictResponse(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("method")] string Method,
    [property: JsonPropertyName("dataset")] string Dataset,
    [property: JsonPropertyName("input")] string Input,
    [property: JsonPropertyName("scores")] string Scores,
    [property: JsonPropertyName("classes")] int Classes,
    [property: JsonPropertyName("predictions")] IReadOnlyList<Prediction> Predictions);

/// <summary>Response of <c>GET /health</c>.</summary>
public sealed record HealthResponse(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("device")] IReadOnlyDictionary<string, object> Device,
    [property: JsonPropertyName("embed_url")] string? EmbedUrl);

internal sealed record PredictRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("texts"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] IReadOnlyList<string>? Texts,
    [property: JsonPropertyName("embeddings"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] IReadOnlyList<float[]>? Embeddings,
    [property: JsonPropertyName("top_k")] int TopK,
    [property: JsonPropertyName("distribution")] bool Distribution);

internal sealed record ModelsResponse([property: JsonPropertyName("models")] IReadOnlyList<ModelInfo> Models);

internal sealed record EmbeddingsRequest(
    [property: JsonPropertyName("input")] IReadOnlyList<string> Input,
    [property: JsonPropertyName("max_tokens")] int MaxTokens);

internal sealed record EmbeddingItem(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("embedding")] float[] Embedding);

internal sealed record EmbeddingsResponse([property: JsonPropertyName("data")] IReadOnlyList<EmbeddingItem> Data);

internal sealed record ErrorResponse([property: JsonPropertyName("error")] string Error);
