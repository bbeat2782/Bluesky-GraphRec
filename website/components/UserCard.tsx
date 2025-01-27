import Linkify from 'linkify-react';

interface UserCardProps {
  profile: {
    avatar?: string;
    displayName?: string;
    handle: string;
    description?: string;
  };
}

export default function UserCard({ profile }: UserCardProps) {
  // Custom options for linkify
  const linkifyOptions = {
    target: '_blank',
    className: 'text-blue-400 hover:text-blue-300 underline',
    rel: 'noopener noreferrer'
  };

  return (
    <div className="w-[300px] h-[160px] rounded-lg overflow-hidden shadow-lg bg-gray-800 p-6 text-white">
      <div className="flex items-center space-x-4">
        <img
          src={profile.avatar || '/default-avatar.png'}
          alt={profile.displayName || profile.handle}
          className="w-16 h-16 rounded-full"
        />
        <div>
          <h2 className="text-xl font-bold text-white">{profile.displayName}</h2>
          <p className="text-gray-400">@{profile.handle}</p>
        </div>
      </div>
      {profile.description && (
        <p className="mt-4 text-gray-300">
          <Linkify options={linkifyOptions}>{profile.description}</Linkify>
        </p>
      )}
    </div>
  );
} 