using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

namespace EmbeddingsBench.Client;

/// <summary>Registration for ASP.NET Core / generic-host applications.</summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers <see cref="IEmbeddingsBenchClient"/> as a typed HttpClient (IHttpClientFactory).
    /// <code>builder.Services.AddEmbeddingsBench(o => { o.BaseAddress = new("http://gpu-host:8096"); o.ApiKey = "…"; });</code>
    /// </summary>
    public static IHttpClientBuilder AddEmbeddingsBench(this IServiceCollection services, Action<EmbeddingsBenchOptions> configure)
    {
        services.Configure(configure);
        return services.AddHttpClient<IEmbeddingsBenchClient, EmbeddingsBenchClient>((sp, http) =>
            EmbeddingsBenchClient.Configure(http, sp.GetRequiredService<IOptions<EmbeddingsBenchOptions>>().Value));
    }
}
