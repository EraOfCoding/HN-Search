import React, { useState } from 'react';
import { Search, ExternalLink, Loader2, AlertCircle } from 'lucide-react';

export default function HNSearch() {
    const [prompt, setPrompt] = useState('');
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSearch = async () => {
        if (!prompt.trim()) {
            setError('Please enter a search query');
            return;
        }

        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const response = await fetch('http://localhost:8000/search-stories', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    limit: 10
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setResults(data);
        } catch (err) {
            setError(err.message || 'Failed to fetch results. Make sure the API server is running.');
            console.error('Search error:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !loading) {
            handleSearch();
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-orange-50 to-orange-100 p-8">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-orange-600 mb-2">
                        HackerNews Semantic Search
                    </h1>
                    <p className="text-gray-600">
                        Search top 500 HN stories using AI-powered semantic similarity
                    </p>
                </div>

                {/* Search Input */}
                <div className="mb-8">
                    <div className="bg-white rounded-lg shadow-lg p-6">
                        <div className="flex gap-3">
                            <div className="flex-1">
                                <input
                                    type="text"
                                    value={prompt}
                                    onChange={(e) => setPrompt(e.target.value)}
                                    onKeyPress={handleKeyPress}
                                    placeholder="Enter your search query (e.g., 'machine learning frameworks')"
                                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent outline-none"
                                    disabled={loading}
                                />
                            </div>
                            <button
                                onClick={handleSearch}
                                disabled={loading}
                                className="px-6 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
                            >
                                {loading ? (
                                    <>
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                        Searching...
                                    </>
                                ) : (
                                    <>
                                        <Search className="w-5 h-5" />
                                        Search
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Error Message */}
                {error && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start gap-3">
                        <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                        <div>
                            <h3 className="font-semibold text-red-800 mb-1">Error</h3>
                            <p className="text-red-600 text-sm">{error}</p>
                        </div>
                    </div>
                )}

                {/* Loading State */}
                {loading && (
                    <div className="bg-white rounded-lg shadow-lg p-12 text-center">
                        <Loader2 className="w-12 h-12 animate-spin text-orange-600 mx-auto mb-4" />
                        <p className="text-gray-600">
                            Searching through 500 stories and computing similarities...
                        </p>
                        <p className="text-sm text-gray-500 mt-2">
                            This may take a minute or two
                        </p>
                    </div>
                )}

                {/* Results */}
                {results && !loading && (
                    <div className="space-y-4">
                        {/* Results Header */}
                        <div className="bg-white rounded-lg shadow-md p-4">
                            <p className="text-gray-700">
                                Found <span className="font-bold text-orange-600">{results.results.length}</span> most relevant stories
                                {results.query && (
                                    <span className="text-gray-500"> for "{results.query}"</span>
                                )}
                            </p>
                            <p className="text-sm text-gray-500 mt-1">
                                Searched {results.total_stories_searched} stories
                            </p>
                        </div>

                        {/* Results List */}
                        {results.results.map((result, index) => (
                            <div
                                key={index}
                                className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow"
                            >
                                <div className="flex items-start justify-between gap-4 mb-3">
                                    <h3 className="text-lg font-semibold text-gray-800 flex-1">
                                        {index + 1}. {result.title}
                                    </h3>
                                    <div className="flex items-center gap-2 flex-shrink-0">
                                        <span className="px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-sm font-medium">
                                            {(result.similarity * 100).toFixed(1)}% match
                                        </span>
                                        {result.score && (
                                            <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-sm">
                                                {result.score} pts
                                            </span>
                                        )}
                                    </div>
                                </div>

                                {result.text_preview && (
                                    <p className="text-gray-600 text-sm mb-4 line-clamp-2">
                                        {result.text_preview}...
                                    </p>
                                )}

                                <div className="flex gap-3">
                                    {result.url && (
                                        <a
                                            href={result.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="inline-flex items-center gap-2 text-orange-600 hover:text-orange-700 font-medium text-sm"
                                        >
                                            <ExternalLink className="w-4 h-4" />
                                            View Article
                                        </a>
                                    )}
                                    <a
                                        href={result.hn_url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-700 font-medium text-sm"
                                    >
                                        <ExternalLink className="w-4 h-4" />
                                        HN Discussion
                                    </a>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Empty State */}
                {results && results.results.length === 0 && !loading && (
                    <div className="bg-white rounded-lg shadow-lg p-12 text-center">
                        <Search className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <h3 className="text-lg font-semibold text-gray-700 mb-2">
                            No results found
                        </h3>
                        <p className="text-gray-500">
                            Try a different search query
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}