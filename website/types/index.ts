export interface Profile {
  avatar?: string;
  displayName?: string;
  handle: string;
  description?: string;
  follows?: string[]; // Array of handles/DIDs that this user follows
} 