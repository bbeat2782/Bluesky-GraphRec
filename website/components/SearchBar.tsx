import { useState } from 'react';

interface SearchBarProps {
  onSearch: (query: string) => void;
}

export default function SearchBar({ onSearch }: SearchBarProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(query);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-md">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter DID, AT URI, or bsky.app URL"
        className="w-full px-4 py-2 bg-gray-800 text-white border border-gray-700 rounded-lg 
                 shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                 placeholder-gray-400"
      />
    </form>
  );
} 