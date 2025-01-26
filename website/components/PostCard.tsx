import { format } from 'date-fns';
import Linkify from 'linkify-react';
import type { Post } from '@/utils/api';

interface PostCardProps {
  post: Post;
}

export default function PostCard({ post }: PostCardProps) {
  const linkifyOptions = {
    target: '_blank',
    className: 'text-blue-400 hover:text-blue-300 underline',
    rel: 'noopener noreferrer'
  };

  return (
    <div className="p-4 bg-gray-700/50 rounded-lg">
      <div className="flex items-center gap-3 mb-2">
        <img
          src={post.author.avatar || '/default-avatar.png'}
          alt={post.author.displayName || post.author.handle}
          className="w-10 h-10 rounded-full"
        />
        <div>
          <div className="font-medium">{post.author.displayName}</div>
          <div className="text-sm text-gray-400">@{post.author.handle}</div>
        </div>
        <div className="ml-auto text-sm text-gray-400">
          {format(new Date(post.createdAt), 'MMM d, yyyy')}
        </div>
      </div>
      <div className="text-gray-100">
        <Linkify options={linkifyOptions}>{post.text}</Linkify>
      </div>
      {post.embed?.images && (
        <div className="mt-2 grid grid-cols-2 gap-2">
          {post.embed.images.map((img, i) => (
            <img
              key={i}
              src={img.thumb}
              alt={img.alt || ''}
              className="rounded-lg max-h-48 object-cover"
            />
          ))}
        </div>
      )}
    </div>
  );
} 