'use client';

import { useState, useEffect } from 'react';
import { format } from 'date-fns';
import type { Profile } from '@/types';
import type { Post, Like, Follow } from '@/utils/api';
import { fetchUserPosts, fetchUserLikes, fetchUserFollows } from '@/utils/api';
import PostCard from '@/components/PostCard';

type Tab = 'posts' | 'likes' | 'follows';

interface DateRange {
  start: Date;
  end: Date;
}

interface UserPanelProps {
  user: Profile;
  onClose: () => void;
}

export default function UserPanel({ user, onClose }: UserPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>('posts');
  const [dateRange, setDateRange] = useState<DateRange>({
    start: new Date('2023-01-01'),
    end: new Date()
  });

  const [posts, setPosts] = useState<Post[]>([]);
  const [likes, setLikes] = useState<Like[]>([]);
  const [follows, setFollows] = useState<Follow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tabs: { id: Tab; label: string }[] = [
    { id: 'posts', label: 'Posts' },
    { id: 'likes', label: 'Likes' },
    { id: 'follows', label: 'Follows' }
  ];

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        switch (activeTab) {
          case 'posts':
            const postsData = await fetchUserPosts(user.handle);
            setPosts(postsData.posts);
            break;
          case 'likes':
            const likesData = await fetchUserLikes(user.handle);
            setLikes(likesData.likes);
            break;
          case 'follows':
            const followsData = await fetchUserFollows(user.handle);
            setFollows(followsData.follows);
            break;
        }
      } catch (err) {
        setError('Failed to load data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [activeTab, user.handle]);

  return (
    <div className="fixed right-0 top-0 h-screen w-[600px] bg-gray-800/95 text-white shadow-xl backdrop-blur-sm overflow-hidden transition-transform">
      {/* Header */}
      <div className="p-6 border-b border-gray-700">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold">{user.displayName}</h2>
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-white"
          >
            ✕
          </button>
        </div>
        <p className="text-gray-400">@{user.handle}</p>
      </div>

      {/* Date Range Picker */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex gap-4 items-center">
          <label className="text-sm text-gray-400">Date Range:</label>
          <input
            type="date"
            value={format(dateRange.start, 'yyyy-MM-dd')}
            onChange={(e) => setDateRange(prev => ({
              ...prev,
              start: new Date(e.target.value)
            }))}
            className="bg-gray-700 px-2 py-1 rounded"
          />
          <span className="text-gray-400">to</span>
          <input
            type="date"
            value={format(dateRange.end, 'yyyy-MM-dd')}
            onChange={(e) => setDateRange(prev => ({
              ...prev,
              end: new Date(e.target.value)
            }))}
            className="bg-gray-700 px-2 py-1 rounded"
          />
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-700">
        <nav className="flex">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-3 text-sm font-medium ${
                activeTab === tab.id
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:text-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      <div className="p-6 overflow-y-auto" style={{ height: 'calc(100vh - 240px)' }}>
        {loading ? (
          <div className="flex justify-center items-center h-full">
            <div className="text-gray-400">Loading...</div>
          </div>
        ) : error ? (
          <div className="text-red-400">{error}</div>
        ) : (
          <div className="space-y-4">
            {activeTab === 'posts' && posts.map(post => (
              <PostCard key={post.uri} post={post} />
            ))}
            
            {activeTab === 'likes' && (
              <div className="space-y-4">
                {likes.length === 0 ? (
                  <div className="text-gray-400 text-center p-4">
                    Likes are not accessible for this profile
                  </div>
                ) : (
                  likes.map(like => (
                    <PostCard key={like.uri} post={like.post} />
                  ))
                )}
              </div>
            )}
            
            {activeTab === 'follows' && (
              <div className="grid grid-cols-2 gap-4">
                {follows.map(follow => (
                  <div key={follow.handle} className="flex items-center gap-3 p-3 bg-gray-700/50 rounded-lg">
                    <img
                      src={follow.avatar || '/default-avatar.png'}
                      alt={follow.displayName || follow.handle}
                      className="w-10 h-10 rounded-full"
                    />
                    <div>
                      <div className="font-medium">{follow.displayName}</div>
                      <div className="text-sm text-gray-400">@{follow.handle}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
} 