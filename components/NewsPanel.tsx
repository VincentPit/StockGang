"use client";

import type { NewsItem } from "@/lib/api";
import { ExternalLink } from "lucide-react";

interface Props {
  items: NewsItem[];
  loading?: boolean;
}

function timeAgo(ts: string): string {
  const diff = (Date.now() - new Date(ts).getTime()) / 1000;
  if (diff < 60)   return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400)return `${Math.floor(diff / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
}

export default function NewsPanel({ items, loading }: Props) {
  if (loading) return (
    <div className="space-y-2 animate-pulse">
      {[1, 2, 3].map((i) => (
        <div key={i} className="h-16 bg-gray-800 rounded-lg" />
      ))}
    </div>
  );

  if (!items.length) return (
    <div className="text-gray-500 text-sm text-center py-8">No news available</div>
  );

  return (
    <ul className="space-y-2 max-h-96 overflow-y-auto pr-1">
      {items.map((item, i) => (
        <li key={i} className="bg-gray-900 rounded-lg p-3 flex flex-col gap-1 hover:bg-gray-800 transition-colors">
          <div className="flex items-start justify-between gap-2">
            <p className="text-sm text-gray-200 leading-snug font-medium flex-1 min-w-0 line-clamp-2">
              {item.title}
            </p>
            {item.url && (
              <a
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-shrink-0 text-sky-400 hover:text-sky-300"
              >
                <ExternalLink className="w-3.5 h-3.5 mt-0.5" />
              </a>
            )}
          </div>
          {item.content && (
            <p className="text-xs text-gray-500 line-clamp-1">{item.content}</p>
          )}
          <div className="flex gap-2 text-xs text-gray-600">
            {item.source && <span>{item.source}</span>}
            <span>{timeAgo(item.ts)}</span>
          </div>
        </li>
      ))}
    </ul>
  );
}
