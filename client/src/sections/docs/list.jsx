import React from "react";
import { cn } from "@/lib/utils";
export default function List({ children, className }) {
  return (
    <ul
      className={cn(
        "space-y-4 pl-12 w-full font-biryani text-gray-200 text-sm leading-56 list-disc",
        className
      )}
    >
      {children}
    </ul>
  );
}
