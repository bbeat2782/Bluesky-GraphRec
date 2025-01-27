import { AtpAgent } from '@atproto/api';

let agent: AtpAgent | null = null;
let authPromise: Promise<AtpAgent> | null = null;

async function getAuthenticatedAgent(): Promise<AtpAgent> {
  if (!agent || !authPromise) {
    agent = new AtpAgent({ service: 'https://bsky.social' });
    authPromise = agent.login({
      identifier: process.env.NEXT_PUBLIC_BSKY_USERNAME!,
      password: process.env.NEXT_PUBLIC_BSKY_PASSWORD!,
    }).then(() => agent!);
  }
  return authPromise;
}

async function retryWithAuth<T>(fn: () => Promise<T>): Promise<T> {
  try {
    await getAuthenticatedAgent();
    return await fn();
  } catch (error: any) {
    if (error?.message?.includes('Authentication Required')) {
      // Reset auth and try once more
      agent = null;
      authPromise = null;
      await getAuthenticatedAgent();
      return await fn();
    }
    throw error;
  }
}

export interface Post {
  uri: string;
  cid: string;
  text: string;
  createdAt: string;
  author: {
    handle: string;
    displayName?: string;
    avatar?: string;
  };
  embed?: {
    images?: { thumb?: string; fullsize?: string; alt?: string }[];
    external?: { uri: string; title: string; description: string };
  };
}

export interface Like {
  uri: string;
  createdAt: string;
  post: Post;
}

export interface Follow {
  handle: string;
  displayName?: string;
  avatar?: string;
  description?: string;
}

export async function fetchUserPosts(
  actor: string,
  limit: number = 50,
  cursor?: string
): Promise<{ posts: Post[]; cursor?: string }> {
  return retryWithAuth(async () => {
    const agent = await getAuthenticatedAgent();
    const response = await agent.api.app.bsky.feed.getAuthorFeed({
      actor,
      limit,
      cursor,
    });

    const posts = response.data.feed.map(item => ({
      uri: item.post.uri,
      cid: item.post.cid,
      text: item.post.record.text,
      createdAt: item.post.record.createdAt,
      author: {
        handle: item.post.author.handle,
        displayName: item.post.author.displayName,
        avatar: item.post.author.avatar,
      },
      embed: item.post.embed,
    }));

    return {
      posts,
      cursor: response.data.cursor,
    };
  });
}

export async function fetchUserLikes(
  actor: string,
  limit: number = 50,
  cursor?: string
): Promise<{ likes: Like[]; cursor?: string }> {
  return retryWithAuth(async () => {
    const agent = await getAuthenticatedAgent();
    try {
      const response = await agent.api.app.bsky.feed.getActorLikes({
        actor,
        limit,
        cursor,
      });

      const likes = response.data.feed.map(item => ({
        uri: item.uri,
        createdAt: item.record.createdAt,
        post: {
          uri: item.post.uri,
          cid: item.post.cid,
          text: item.post.record.text,
          createdAt: item.post.record.createdAt,
          author: {
            handle: item.post.author.handle,
            displayName: item.post.author.displayName,
            avatar: item.post.author.avatar,
          },
          embed: item.post.embed,
        },
      }));

      return {
        likes,
        cursor: response.data.cursor,
      };
    } catch (error: any) {
      console.log('Likes fetch error:', {
        error,
        message: error?.message,
        status: error?.status,
        actor,
      });

      // Handle various error cases
      if (error?.message?.includes('Profile not found') || 
          error?.status === 400 ||
          error?.message?.includes('Bad Request')) {
        console.log('Profile likes are not accessible');
        return { likes: [] };
      }
      throw error;
    }
  });
}

export async function fetchUserFollows(
  actor: string,
  limit: number = 100,
  cursor?: string
): Promise<{ follows: Follow[]; cursor?: string }> {
  const agent = await getAuthenticatedAgent();
  const response = await agent.api.app.bsky.graph.getFollows({
    actor,
    limit,
    cursor,
  });

  const follows = response.data.follows.map(follow => ({
    handle: follow.handle,
    displayName: follow.displayName,
    avatar: follow.avatar,
    description: follow.description,
  }));

  return {
    follows,
    cursor: response.data.cursor,
  };
} 