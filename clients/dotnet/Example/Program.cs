// Example: classify texts with an EmbeddingsBench model from a .NET application (generic host + DI).
//   EB_INFER_URL=http://gpu-host:8096 EB_API_KEY=secret dotnet run --project Example -- banking-logreg "My card has not arrived yet"
using EmbeddingsBench.Client;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = Host.CreateApplicationBuilder(args);
builder.Services.AddEmbeddingsBench(o =>
{
    o.BaseAddress = new Uri(Environment.GetEnvironmentVariable("EB_INFER_URL") ?? "http://localhost:8096");
    o.ApiKey = Environment.GetEnvironmentVariable("EB_API_KEY");
});
using var host = builder.Build();
var client = host.Services.GetRequiredService<IEmbeddingsBenchClient>();

var health = await client.GetHealthAsync();
Console.WriteLine($"service: {health.Status}, device {string.Join(", ", health.Device.Select(kv => $"{kv.Key}={kv.Value}"))}");

var models = await client.ListModelsAsync();
Console.WriteLine($"{models.Count} models, e.g.:");
foreach (var m in models.Take(5))
    Console.WriteLine($"  {m.Model,-32} {m.Method,-18} {m.Dataset,-20} input={m.Input,-16} test={m.TestAccuracy:P1}");

var model = args.Length > 0 ? args[0] : "banking-logreg";
var texts = args.Length > 1 ? args[1..] : new[] { "My card has not arrived yet", "How do I change my PIN?" };
try
{
    var result = await client.PredictAsync(model, texts, topK: 3);
    Console.WriteLine($"{result.Model} ({result.Method}, {result.Scores}):");
    foreach (var (text, p) in texts.Zip(result.Predictions))
        Console.WriteLine($"  \"{text}\" -> {p.Label} ({p.Confidence:P1}); top 3: {string.Join(", ", p.TopK.Select(t => $"{t.Label} {t.Probability:P0}"))}");
}
catch (EmbeddingsBenchException e)
{
    Console.WriteLine($"error {e.StatusCode}: {e.Message}");
}
