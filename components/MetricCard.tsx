import clsx from "clsx";

interface Props {
  label: string;
  value: string;
  positive?: boolean;
}

export function MetricCard({ label, value, positive }: Props) {
  return (
    <div className="bg-gray-800 rounded-xl p-4">
      <p className="text-xs text-gray-400 mb-1">{label}</p>
      <p
        className={clsx(
          "text-xl font-bold",
          positive === true && "text-emerald-400",
          positive === false && "text-rose-400",
          positive === undefined && "text-white"
        )}
      >
        {value}
      </p>
    </div>
  );
}
