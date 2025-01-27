'use client';

import { useState } from 'react';
import { AtpAgent } from '@atproto/api';
import SearchBar from '@/components/SearchBar';
import GraphCanvas from '@/components/GraphCanvas';
import type { Profile } from '@/types';

export type Node = Profile & {
  x: number;
  y: number;
  fx?: number;
  fy?: number;
  follows?: string[];
};

const agent = new AtpAgent({ service: 'https://bsky.social' });

// You can create a test account at bsky.app and use those credentials
// Or use an App Password from your account settings
await agent.login({
  identifier: process.env.NEXT_PUBLIC_BSKY_USERNAME!,
  password: process.env.NEXT_PUBLIC_BSKY_PASSWORD!,
});

export default function Home() {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSearch = async (query: string) => {
    try {
      setError(null);
      setIsLoading(true);
      
      let handle = query;
      if (query.includes('bsky.app/profile/')) {
        handle = query.split('/profile/')[1];
      }
      if (query.startsWith('at://')) {
        handle = query.split('/')[2];
      }

      // Get profile
      const profileResponse = await agent.getProfile({ actor: handle });
      
      // Get follows
      const followsResponse = await agent.getFollows({ actor: handle });
      const follows = followsResponse.data.follows.map(f => f.handle);

      const newNode: Node = {
        ...profileResponse.data,
        follows,
        x: window.innerWidth / 2 + (Math.random() - 0.5) * 200,
        y: window.innerHeight / 2 + (Math.random() - 0.5) * 200,
      };

      setNodes(prev => [...prev, newNode]);
    } catch (error) {
      console.error('Error fetching user profile:', error);
      setError('Failed to fetch user profile. Please check the input and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="w-full h-screen fixed inset-0">
      <div className="fixed top-4 left-4 z-10 w-96">
        <SearchBar onSearch={handleSearch} />
        {error && (
          <div className="text-red-500 mt-2">
            {error}
          </div>
        )}
        {isLoading && (
          <div className="text-white mt-2">
            Loading...
          </div>
        )}
      </div>
      <GraphCanvas nodes={nodes} setNodes={setNodes} />
    </main>
  );
}
