import { humanize } from "../lib/format";

/** Rows = true label, columns = predicted label. Correct cells teal, mistakes coral. */
export function ConfusionMatrix({ cm }: { cm: { labels: string[]; matrix: number[][]; unparseable?: number[] | null } }) {
  const max = Math.max(1, ...cm.matrix.flat());
  const many = cm.labels.length > 8;
  const cell = many ? "h-7 w-10 text-[11px]" : "h-9 w-16";
  return (
    <div className="overflow-x-auto">
      <table className="text-[12px] border-separate border-spacing-0.5">
        <caption className="caption-bottom pt-2 text-left text-faint">Rows: true label. Columns: predicted label.{cm.unparseable ? " “?” = answer that matched no label." : ""}</caption>
        <thead>
          <tr>
            <th />
            {cm.labels.map((l) => <th key={l} className="px-1 font-normal text-muted align-bottom"><div className={many ? "w-10 truncate" : "w-16 truncate"} title={l}>{humanize(l)}</div></th>)}
            {cm.unparseable && <th className="px-1 font-normal text-muted align-bottom">?</th>}
          </tr>
        </thead>
        <tbody>
          {cm.matrix.map((row, i) => (
            <tr key={cm.labels[i]}>
              <th className="pr-2 text-right font-normal text-muted whitespace-nowrap max-w-[160px] truncate" title={cm.labels[i]}>{humanize(cm.labels[i])}</th>
              {row.map((v, j) => {
                const a = v / max;
                const diag = i === j;
                return (
                  <td key={j} className={`num rounded text-center ${cell}`} style={{ background: v ? (diag ? `rgba(87,194,176,${0.15 + a * 0.6})` : `rgba(238,116,104,${0.2 + a * 0.6})`) : "#19212c", color: v ? "#e4e8ee" : "#66738a" }}>{v}</td>
                );
              })}
              {cm.unparseable && <td className={`num rounded text-center ${cell}`} style={{ background: cm.unparseable[i] ? "rgba(162,139,235,0.45)" : "#19212c", color: cm.unparseable[i] ? "#e4e8ee" : "#66738a" }}>{cm.unparseable[i]}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
