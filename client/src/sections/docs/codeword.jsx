import { cn } from "@/lib/utils";
import React from "react";

export default function CodeWord({ children, classNames, bg }) {
  return (
    <span>
      <code
        className={cn(
          " p-1 rounded-md text-red-700 font-bold text-lg",
          classNames,
          bg && "bg-white/10 font-normal text-md"
        )}
      >
        {children}
      </code>
    </span>
  );
}
